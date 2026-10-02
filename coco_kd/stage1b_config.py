"""Opt-in configuration; historical Stage 1 JSONs remain unchanged."""
import json
from dataclasses import dataclass
from pathlib import Path

from .config import Config


@dataclass(kw_only=True)
class Stage1bConfig(Config):
    train_augmentation: str
    stage1b_run: str | None = None

    @classmethod
    def load(cls, path=None, **overrides):
        values = json.loads(Path(path).read_text()) if path else overrides
        if "train_augmentation" not in values:
            raise ValueError("train_augmentation must be explicit; no default")
        if path and "temperature" not in values:
            raise ValueError("Stage 1b temperature must be explicit in JSON")
        cfg = super().load(path, **overrides)
        if cfg.train_augmentation not in ("flip", "rrc", "rrc_mix"):
            raise ValueError("Invalid train_augmentation")
        if cfg.stage1_run is not None:
            raise ValueError("Stage 1b config cannot target Stage 1 outputs")
        return cfg
