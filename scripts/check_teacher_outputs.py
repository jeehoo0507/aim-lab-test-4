"""Measure unmasked teachers on augmented train once and validation; never test."""
import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from coco_kd.data import CocoSubset, loader
from coco_kd.models import build_model
from coco_kd.stage1 import (MANIFEST_SHA256, OUTPUT, check_assets, read_config, require_tests,
                            teacher_pass, teacher_specs)
from coco_kd.utils import autocast, load_checkpoint, seed_all, setup_device, sha256, source_fingerprint, write_json


def kl_to_ls(probabilities, labels):
    """Per-example KL(p || q), nats; q_y=.91, q_other=.01 for COCO-10."""
    p = probabilities.double()
    if p.ndim != 2 or p.shape[1] != 10 or not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError("Expected finite nonnegative COCO-10 probabilities")
    if not torch.allclose(p.sum(1), torch.ones(len(p), dtype=p.dtype, device=p.device), atol=1e-6, rtol=0):
        raise ValueError("Probabilities must sum to one")
    q = torch.full_like(p, 0.01)
    q.scatter_(1, labels[:, None], 0.91)
    return (torch.special.xlogy(p, p) - p * q.log()).sum(1)


def nontarget_entropy(logits, labels, temperature):
    """Exclude the true class BEFORE softmax, avoiding 1-p_y cancellation."""
    keep = torch.ones_like(logits, dtype=torch.bool)
    keep.scatter_(1, labels[:, None], False)
    non_target = logits.double()[keep].reshape(len(logits), 9) / temperature
    log_p = non_target.log_softmax(1)
    return -(log_p.exp() * log_p).sum(1) / math.log(9)


def validate_checkpoint(state, seed, name, expected_ls):
    if state["role"] != "teacher" or state["metadata_sha256"] != MANIFEST_SHA256 or state["config"]["seed"] != seed:
        raise ValueError(f"Teacher checkpoint provenance mismatch: {name}")
    if state["config"]["teacher_label_smoothing"] != expected_ls:
        raise ValueError(f"Teacher LS mismatch: {name}")
    if name == "T_LS01_best":
        if seed != 0 or state["config"]["stage1_run"] != "T_LS01" or not 1 <= state["epoch"] <= 30:
            raise ValueError("Expected seed 0 T_LS01 best checkpoint from epochs 1..30")
    elif state["epoch"] != 30:
        raise ValueError(f"Expected epoch 30 last teacher: {name}")


@torch.inference_mode()
def measure(model, cfg, split, device):
    if split not in ("train", "val"):
        raise ValueError("Teacher output check permits train and val only")
    dataset = CocoSubset(cfg.data_root, split, train=split == "train", threshold=cfg.foreground_threshold)
    # Reset after model construction: every teacher sees identical augmented train inputs.
    seed_all(0)
    measurement_cfg = replace(cfg, seed=0)
    total = correct = 0
    class_counts = torch.zeros(10, dtype=torch.int64)
    class_correct = torch.zeros(10, dtype=torch.int64)
    sums = {"kl_to_ls": 0.0, "nontarget_entropy@T1": 0.0, "nontarget_entropy@T4": 0.0}
    for batch in loader(dataset, measurement_cfg, train=split == "train", epoch=0):
        labels = batch["label"]
        with autocast(cfg, device):
            logits = model(batch["image"].to(device)).float().cpu()
        hit = logits.argmax(1) == labels
        total += len(labels)
        correct += hit.sum().item()
        class_counts += torch.bincount(labels, minlength=10)
        class_correct += torch.bincount(labels[hit], minlength=10)
        sums["kl_to_ls"] += kl_to_ls(logits.double().softmax(1), labels).sum().item()
        for t in (1, 4):
            sums[f"nontarget_entropy@T{t}"] += nontarget_entropy(logits, labels, t).sum().item()
    if (class_counts == 0).any():
        raise ValueError(f"Missing class in {split}")
    return {"samples": total, "accuracy": correct / total,
            "macro_accuracy": (class_correct.double() / class_counts).mean().item(),
            **{key: value / total for key, value in sums.items()}}


def check_teachers(seed):
    require_tests()
    check_assets()
    cfg = read_config(seed, "T_LS0")
    device = setup_device(cfg)
    record = {"seed": seed, "measurement_seed": 0, "manifest_sha256": MANIFEST_SHA256,
              "source_sha256": source_fingerprint(), "gate_split": "train",
              "train_transform": "full_image_resize224_flip_train_only", "teachers": {}}
    for name, path, expected_ls in teacher_specs(seed):
        state = load_checkpoint(path)
        validate_checkpoint(state, seed, name, expected_ls)
        model = build_model("teacher", cfg).to(device).requires_grad_(False).eval()
        model.load_state_dict(state["model"])
        del state
        row = {"checkpoint": str(path), "checkpoint_sha256": sha256(path),
               "teacher_label_smoothing": expected_ls,
               "train": measure(model, cfg, "train", device), "val": measure(model, cfg, "val", device)}
        row["gate"] = "N/A" if name == "T_LS01_best" else "PASS" if teacher_pass(name, row["train"]["kl_to_ls"]) else "FAIL"
        record["teachers"][name] = row
        del model
    write_json(OUTPUT / f"seed_{seed}" / "teacher_checks.json", record)
    if any(row["gate"] == "FAIL" for row in record["teachers"].values()):
        raise ValueError("Teacher output check failed; stop and ask. Results saved; no student may start.")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    check_teachers(parser.parse_args().seed)
