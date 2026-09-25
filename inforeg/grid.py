"""The benchmark grid: one entry per regularizer configuration, crossed with n_train and seeds."""
from __future__ import annotations

from .config import Config

GRID = [
    dict(method="none"),
    dict(method="weight_decay", weight_decay=0.1),
    dict(method="weight_decay", weight_decay=1.0),   # wd=0.1 is nearly inert at lr=2e-4 over ~300 steps
    dict(method="label_smoothing", epsilon=0.1),
    dict(method="confidence_penalty", beta=0.05),
    dict(method="confidence_penalty", beta=0.2),
    dict(method="vib", beta=1e-4),
    dict(method="vib", beta=1e-3),
    dict(method="vib", beta=1e-2),
]


def grid_configs(n_train: list[int], seeds: list[int], **overrides) -> list[Config]:
    """GRID x n_train x seeds; `overrides` (epochs, model_name, ...) apply to every run."""
    return [Config(**g, n_train=n, seed=s, **overrides) for n in n_train for s in seeds for g in GRID]
