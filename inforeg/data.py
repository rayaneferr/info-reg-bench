"""MNLI (in-domain, 3 classes) and HANS (out-of-domain, 2 classes) loaders."""
from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

# GLUE/MNLI labels: 0 entailment, 1 neutral, 2 contradiction
# HANS labels:      0 entailment, 1 non-entailment
MNLI_LABELS = ["entailment", "neutral", "contradiction"]
HANS_LABELS = ["entailment", "non-entailment"]


def load_mnli(n_train: int, n_eval: int, seed: int):
    ds = load_dataset("nyu-mll/glue", "mnli")
    train = ds["train"].shuffle(seed=seed).select(range(n_train))
    val = ds["validation_matched"].shuffle(seed=seed).select(range(n_eval))
    return train, val


def load_hans(n_hans: int, seed: int):
    ds = load_dataset("jhu-cogsci/hans", revision="refs/convert/parquet")["validation"]
    if n_hans and n_hans < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_hans))
    return ds


def tokenize_pairs(ds, tokenizer, max_length: int):
    def _tok(batch):
        enc = tokenizer(
            batch["premise"], batch["hypothesis"],
            truncation=True, max_length=max_length,
        )
        enc["labels"] = batch["label"]
        return enc

    keep = {"input_ids", "attention_mask", "labels"}
    cols = [c for c in ds.column_names if c not in keep]
    return ds.map(_tok, batched=True, remove_columns=cols)


def make_loader(ds, tokenizer, batch_size: int, shuffle: bool, seed: int = 0):
    collator = DataCollatorWithPadding(tokenizer, padding=True)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collator,
        generator=gen if shuffle else None, num_workers=0,
    )
