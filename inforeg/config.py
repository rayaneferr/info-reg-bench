from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

METHODS = ("none", "weight_decay", "label_smoothing", "confidence_penalty", "vib")


@dataclass
class Config:
    # --- data ---
    n_train: int = 1000
    n_eval: int = 2000            # MNLI validation_matched subsample
    n_hans: int = 3000            # HANS subsample (OOD)
    max_length: int = 128
    # --- model ---
    model_name: str = "Qwen/Qwen2.5-0.5B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    z_dim: int = 128              # VIB bottleneck size
    # --- regularization ---
    method: str = "none"
    beta: float = 0.0             # strength for confidence_penalty / vib
    epsilon: float = 0.0          # label smoothing
    weight_decay: float = 0.0
    dropout: float = 0.0          # dropout before the classification head
    # --- optim ---
    epochs: int = 5
    batch_size: int = 16
    lr: float = 2e-4
    warmup_ratio: float = 0.06
    seed: int = 0
    # --- hardware ---
    device: str = "auto"          # auto | mps | cuda | cpu
    # --- io ---
    out_dir: str = "results"
    run_name: str = ""
    tags: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if not self.run_name:
            parts = [self.method, f"n{self.n_train}"]
            if self.method in ("confidence_penalty", "vib"):
                parts.append(f"b{self.beta:g}")
            if self.method == "label_smoothing":
                parts.append(f"e{self.epsilon:g}")
            if self.method == "weight_decay":
                parts.append(f"wd{self.weight_decay:g}")
            parts.append(f"s{self.seed}")
            self.run_name = "_".join(parts)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), indent=2))
