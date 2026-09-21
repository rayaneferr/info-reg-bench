"""Qwen2.5 backbone + LoRA + classification head (linear or Variational Information Bottleneck)."""
from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from .config import Config


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Classifier(nn.Module):
    """Pools the last non-pad token of a causal LM and classifies it.

    head="linear": pooled -> Dropout -> Linear                     (baseline)
    head="vib":    pooled -> MLP -> (mu, logvar) -> z ~ N(mu, s^2) -> Linear
                   and returns KL( q(z|x) || N(0, I) ), the variational upper bound
                   on I(X; Z) used by the Information Bottleneck regularizer.
    """

    def __init__(self, backbone: nn.Module, hidden: int, num_labels: int,
                 head: str = "linear", z_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.head_type = head
        # shared by both heads: keeps the comparison fair and tames Qwen's large hidden norms
        self.norm = nn.LayerNorm(hidden)
        if head == "linear":
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_labels))
        elif head == "vib":
            self.enc = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 2 * z_dim)
            )
            self.cls = nn.Linear(z_dim, num_labels)
        else:
            raise ValueError(head)

    def pool(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state                                  # [B, T, H]
        last = attention_mask.sum(dim=1) - 1                       # right padding
        return h[torch.arange(h.size(0), device=h.device), last]  # [B, H]

    def forward(self, input_ids, attention_mask, **_):
        pooled = self.norm(self.pool(input_ids, attention_mask))
        if self.head_type == "linear":
            logits = self.head(pooled)
            kl = logits.new_zeros(())
            return {"logits": logits, "kl": kl, "pooled": pooled}
        mu, logvar = self.enc(pooled).chunk(2, dim=-1)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        # KL(N(mu, sigma^2) || N(0, I)) per example, averaged over batch
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(dim=-1).mean()
        return {"logits": self.cls(z), "kl": kl, "pooled": pooled, "mu": mu}


def build(cfg: Config, num_labels: int = 3):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModel.from_pretrained(cfg.model_name, dtype=torch.float32)
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    )
    backbone = get_peft_model(backbone, lora)

    head = "vib" if cfg.method == "vib" else "linear"
    model = Classifier(backbone, backbone.config.hidden_size, num_labels,
                       head=head, z_dim=cfg.z_dim, dropout=cfg.dropout)
    return model, tokenizer


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
