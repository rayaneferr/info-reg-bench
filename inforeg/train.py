"""Training / evaluation loop. `run(cfg)` trains one configuration and writes results/<run_name>/."""
from __future__ import annotations

import json
import random
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import Config
from .data import load_hans, load_mnli, make_loader, tokenize_pairs
from .losses import compute_loss
from .metrics import mnli_to_hans, summarize
from .model import build, count_trainable, get_device


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, labels, kls = [], [], []
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        logits.append(out["logits"].float().cpu().numpy())
        labels.append(batch["labels"].numpy())
        kls.append(float(out["kl"]))
    return np.concatenate(logits), np.concatenate(labels), float(np.mean(kls))


def run(cfg: Config, force: bool = False) -> dict:
    run_dir = cfg.run_dir
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        print(f"[skip] {cfg.run_name} already done")
        return json.loads(metrics_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    model, tok = build(cfg)
    model.to(device)
    n_tr, n_tot = count_trainable(model)
    print(f"[{cfg.run_name}] device={device} trainable={n_tr/1e6:.2f}M / {n_tot/1e6:.0f}M")

    train_raw, val_raw = load_mnli(cfg.n_train, cfg.n_eval, cfg.seed)
    hans_raw = load_hans(cfg.n_hans, cfg.seed)
    train_ds = tokenize_pairs(train_raw, tok, cfg.max_length)
    val_ds = tokenize_pairs(val_raw, tok, cfg.max_length)
    hans_ds = tokenize_pairs(hans_raw, tok, cfg.max_length)
    train_loader = make_loader(train_ds, tok, cfg.batch_size, shuffle=True, seed=cfg.seed)
    train_eval_loader = make_loader(train_ds, tok, cfg.batch_size * 2, shuffle=False)
    val_loader = make_loader(val_ds, tok, cfg.batch_size * 2, shuffle=False)
    hans_loader = make_loader(hans_ds, tok, cfg.batch_size * 2, shuffle=False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * len(train_loader)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * steps), steps)

    history = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        agg = {}
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        for batch in pbar:
            labels = batch["labels"].to(device)
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss, stats = compute_loss(out, labels, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + v
            pbar.set_postfix(loss=f"{stats['loss']:.3f}", ce=f"{stats['ce']:.3f}")
        agg = {k: v / len(train_loader) for k, v in agg.items()}
        vl, vy, _ = predict(model, val_loader, device)
        v = summarize(vl, vy)
        rec = {"epoch": epoch, **{f"train_{k}": v_ for k, v_ in agg.items()},
               "val_accuracy": v["accuracy"], "val_nll": v["nll"], "val_ece": v["ece"],
               "elapsed_s": time.time() - t0}
        history.append(rec)
        print(f"  ep{epoch} train_ce={agg['ce']:.3f} reg={agg['reg']:.4f} "
              f"val_acc={v['accuracy']:.3f} val_nll={v['nll']:.3f} val_ece={v['ece']:.3f}")

    # ---- final evaluation ----
    tr_l, tr_y, _ = predict(model, train_eval_loader, device)
    va_l, va_y, va_kl = predict(model, val_loader, device)
    ha_l, ha_y, _ = predict(model, hans_loader, device)
    train_m = summarize(tr_l, tr_y)
    val_m = summarize(va_l, va_y)
    hans_m = summarize(mnli_to_hans(ha_l), ha_y)
    # HANS accuracy split by heuristic (lexical_overlap / subsequence / constituent)
    by_heur = {}
    heur = np.array(hans_raw["heuristic"])
    pred2 = mnli_to_hans(ha_l).argmax(-1)
    for h in sorted(set(heur)):
        m = heur == h
        by_heur[h] = float((pred2[m] == ha_y[m]).mean())

    results = {
        "run_name": cfg.run_name, "method": cfg.method, "beta": cfg.beta, "epsilon": cfg.epsilon,
        "weight_decay": cfg.weight_decay, "n_train": cfg.n_train, "seed": cfg.seed,
        "trainable_params": n_tr, "train_time_s": time.time() - t0,
        "train": {k: v for k, v in train_m.items() if k != "reliability"},
        "val": val_m, "hans": hans_m, "hans_by_heuristic": by_heur,
        "generalization_gap_nll": val_m["nll"] - train_m["nll"],
        "generalization_gap_acc": train_m["accuracy"] - val_m["accuracy"],
        "ood_drop_acc": val_m["accuracy"] - hans_m["accuracy"],
        "val_kl": va_kl, "history": history,
    }
    metrics_path.write_text(json.dumps(results, indent=2))
    np.savez_compressed(run_dir / "predictions_val.npz", logits=va_l, labels=va_y)
    np.savez_compressed(run_dir / "predictions_hans.npz", logits=ha_l, labels=ha_y)
    print(f"[done] {cfg.run_name}: val_acc={val_m['accuracy']:.3f} hans_acc={hans_m['accuracy']:.3f} "
          f"val_ece={val_m['ece']:.3f} gap_nll={results['generalization_gap_nll']:.3f} "
          f"({results['train_time_s']/60:.1f} min)")
    return results
