"""Paired RRC and batch mixing, isolated from model RNG consumption.

Batch Mixup/CutMix semantics reference timm/data/mixup.py at
92dbd9e3f9b5d61c4d008223410781da983239fc (Apache-2.0, Ross Wightman).
No runtime timm dependency. Uses reverse-batch pairs and area-corrected lambda.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF

from .data import CocoSubset


def keyed_seed(seed, epoch, item, stream):
    payload = f"stage1b:{seed}:{epoch}:{item}:{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def crop_parameters(image, seed, epoch, sample_id):
    # CPU-only fork; setting the default CPU generator does not reseed CUDA.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(keyed_seed(seed, epoch, sample_id, "crop"))
        box = RandomResizedCrop.get_params(image, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3))
        flip = torch.rand(()).item() < .5
    return box, flip


def paired_crop(image, mask, box, flip):
    image = TF.resized_crop(image, *box, [224, 224], InterpolationMode.BICUBIC, antialias=True)
    mask = TF.resized_crop(mask, *box, [224, 224], InterpolationMode.NEAREST)
    if flip:
        image, mask = TF.hflip(image), TF.hflip(mask)
    return image, mask


class AugmentedCocoSubset(CocoSubset):
    def __init__(self, root, *, seed, threshold=.5):
        super().__init__(root, "train", train=True, threshold=threshold)
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        row = self.records[index]
        for key in ("image", "mask"):
            path = Path(row[key])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Dataset manifest paths must be relative")
        with Image.open(self.root / row["image"]) as file:
            image = file.convert("RGB")
        with Image.open(self.root / row["mask"]) as file:
            mask = file.convert("L")
        if image.size != mask.size:
            raise ValueError("Image/mask dimensions disagree")
        box, flip = crop_parameters(image, self.seed, self.epoch, row["id"])
        image, mask = paired_crop(image, mask, box, flip)
        image = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        coverage = F.avg_pool2d(TF.to_tensor(mask), 16, 16).flatten()
        return {"image": image, "label": row["label"], "sample_id": row["id"],
                "coverage": coverage, "foreground": coverage >= self.threshold,
                "crop_box": torch.tensor(box), "crop_flip": flip}


def mix_parameters(shape, seed, epoch, step):
    rng = np.random.RandomState(keyed_seed(seed, epoch, step, "mix"))
    rng.rand()  # timm probability draw; application probability is fixed to 1.
    cutmix = rng.rand() < .5
    alpha = 1. if cutmix else .8
    lam = float(rng.beta(alpha, alpha))
    box = None
    if cutmix:
        height, width = shape[-2:]
        ratio = math.sqrt(1 - lam)
        cut_h, cut_w = int(height * ratio), int(width * ratio)
        cy, cx = rng.randint(0, height), rng.randint(0, width)
        y0, y1 = max(0, cy - cut_h // 2), min(height, cy + (cut_h + 1) // 2)
        x0, x1 = max(0, cx - cut_w // 2), min(width, cx + (cut_w + 1) // 2)
        box = (int(y0), int(y1), int(x0), int(x1))
        lam = 1 - (y1 - y0) * (x1 - x0) / (height * width)
    return {"kind": "cutmix" if cutmix else "mixup", "lambda": float(lam), "box": box,
            "pairs": list(reversed(range(shape[0])))}


def mix_batch(images, labels, cfg, epoch, step):
    if len(images) % 2:
        raise ValueError("Stage 1b batch mix requires even batch size, including the final batch")
    params = mix_parameters(images.shape, cfg.seed, epoch, step)
    lam, box = params["lambda"], params["box"]
    paired = images.flip(0)
    if box is None:
        mixed = images.clone().mul_(lam).add_(paired.mul(1 - lam))
    else:
        mixed = images.clone()
        y0, y1, x0, x1 = box
        mixed[:, :, y0:y1, x0:x1] = paired[:, :, y0:y1, x0:x1]
    target = F.one_hot(labels, cfg.num_classes).float() * (1 - cfg.label_smoothing)
    target += cfg.label_smoothing / cfg.num_classes
    target = lam * target + (1 - lam) * target.flip(0)
    return mixed, target, params
