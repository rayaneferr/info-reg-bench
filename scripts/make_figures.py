"""Aggregate results/*/metrics.json into results/summary.csv and figures/*.png.

Figures follow a fixed visual grammar: colour encodes the *family* of the regularizer
(where R acts), never the rank; the un-regularised baseline is drawn as a neutral reference.
"""
import json
import sys
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"

# ---- palette (validated, see docs) ---------------------------------------------------------
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FAMILY = {  # where R acts
    "none": ("reference (no regularization)", MUTED),
    "weight_decay": ("weights", "#1baf7a"),
    "label_smoothing": ("outputs", "#eb6834"),
    "confidence_penalty": ("outputs", "#eb6834"),
    "vib": ("representation", "#2a78d6"),
}
# when several methods of the same family share one plot, they need distinct hues
CURVE = {"none": MUTED, "label_smoothing": "#eb6834", "confidence_penalty": "#4a3aa7", "vib": "#2a78d6",
         "weight_decay": "#1baf7a"}
ORDER = ["none", "weight_decay", "label_smoothing", "confidence_penalty", "vib"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "normal", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 2, "lines.markersize": 7,
    "lines.markeredgewidth": 1.6, "lines.markeredgecolor": SURFACE,
})


# ---- data -----------------------------------------------------------------------------------
def load() -> pd.DataFrame:
    rows = []
    for f in sorted(RES.glob("*/metrics.json")):
        r = json.loads(f.read_text())
        rows.append({
            "run": r["run_name"], "method": r["method"], "beta": r["beta"], "epsilon": r["epsilon"],
            "weight_decay": r["weight_decay"], "n_train": r["n_train"], "seed": r["seed"],
            "train_acc": r["train"]["accuracy"], "train_nll": r["train"]["nll"],
            "val_acc": r["val"]["accuracy"], "val_nll": r["val"]["nll"], "val_ece": r["val"]["ece"],
            "val_entropy": r["val"]["entropy"],
            "hans_acc": r["hans"]["accuracy"], "hans_ece": r["hans"]["ece"],
            "gap_nll": r["generalization_gap_nll"], "gap_acc": r["generalization_gap_acc"],
            "ood_drop": r["ood_drop_acc"], "val_kl": r["val_kl"], "time_min": r["train_time_s"] / 60,
            **{f"hans_{k}": v for k, v in r["hans_by_heuristic"].items()},
        })
    if not rows:
        sys.exit("no results found in results/*/metrics.json")
    df = pd.DataFrame(rows)
    df["order"] = df.method.map(ORDER.index)
    df = df.sort_values(["n_train", "order", "weight_decay", "epsilon", "beta", "seed"]).drop(columns="order")
    df.to_csv(RES / "summary.csv", index=False)
    return df


def label(r) -> str:
    if r.method == "vib":
        return f"VIB  β={r.beta:g}"
    if r.method == "confidence_penalty":
        return f"conf. penalty  β={r.beta:g}"
    if r.method == "label_smoothing":
        return f"label smoothing  ε={r.epsilon:g}"
    if r.method == "weight_decay":
        return f"weight decay  λ={r.weight_decay:g}"
    return "no regularization"


def agg(df: pd.DataFrame) -> pd.DataFrame:
    """mean ± std over seeds, one row per configuration, in display order."""
    keys = ["method", "beta", "epsilon", "weight_decay", "n_train"]
    num = df.select_dtypes("number").columns.difference(keys + ["seed"])
    g = df.groupby(keys, sort=False)[list(num)].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()
    g["order"] = g.method.map(ORDER.index)
    g = g.sort_values(["n_train", "order", "weight_decay", "epsilon", "beta"]).drop(columns="order")
    g["label"] = g.apply(label, axis=1)
    g["color"] = g.method.map(lambda m: FAMILY[m][1])
    return g.reset_index(drop=True)


def _ref_line(ax, x, text, y_text=None, color=AXIS):
    ax.axvline(x, color=color, lw=1, zorder=1)
    ax.text(x, y_text if y_text is not None else ax.get_ylim()[1], f" {text}", color=MUTED, fontsize=7.5,
            ha="left", va="bottom", clip_on=False)


# ---- figure 1: overview dot plot -------------------------------------------------------------
def overview(g: pd.DataFrame, n: int):
    sub = g[g.n_train == n].reset_index(drop=True)
    base = sub[sub.method == "none"].iloc[0]
    panels = [("val_acc", "MNLI accuracy  ↑", True), ("hans_acc", "HANS accuracy (out-of-domain)  ↑", True),
              ("val_ece", "Expected calibration error  ↓", False),
              ("gap_nll", "Generalization gap  (val NLL − train NLL)  ↓", False)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.42 * len(sub) + 1.9), sharey=True)
    y = np.arange(len(sub))[::-1]
    for ax, (m, title, higher) in zip(axes, panels, strict=True):
        vals, err = sub[f"{m}_mean"], sub[f"{m}_std"].fillna(0)
        ax.axvline(base[f"{m}_mean"], color=AXIS, lw=1, zorder=1)
        if m == "hans_acc":
            ax.axvline(0.5, color=AXIS, lw=1, zorder=1)
        ax.errorbar(vals, y, xerr=err, fmt="none", ecolor=AXIS, elinewidth=1, capsize=0, zorder=2)
        ax.scatter(vals, y, s=80, c=sub.color, edgecolors=SURFACE, linewidths=1.6, zorder=3)
        best = vals.idxmax() if higher else vals.idxmin()
        ax.annotate(f"{vals[best]:.3f}", (vals[best], y[best]), xytext=(7, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, color=INK)
        ax.set_title(title, loc="left", pad=18)
        ax.grid(axis="y", visible=False)
        lo, hi = vals.min(), vals.max()
        pad = 0.18 * (hi - lo) if hi > lo else 0.02
        ax.set_xlim(min(lo, 0.5 if m == "hans_acc" else lo) - pad, hi + pad * 2.2)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        top = ax.get_ylim()[1]
        ax.text(base[f"{m}_mean"], top, " baseline", color=MUTED, fontsize=7.5, va="bottom", ha="left")
        if m == "hans_acc":
            ax.text(0.5, top, " chance", color=MUTED, fontsize=7.5, va="bottom", ha="right")
    axes[0].set_yticks(y, sub.label)
    # family legend
    seen = {}
    for m in ORDER:
        fam, col = FAMILY[m]
        seen.setdefault(fam, col)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=c, markeredgecolor=SURFACE,
                          label=f if f.startswith("reference") else f"R acts on {f}")
               for f, c in seen.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Same backbone, same LoRA, {n:,} MNLI examples — only the penalty R in  L = CE + R  changes",  # noqa: E501
                 x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="normal")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(FIG / f"overview_n{n}.png", dpi=170)
    plt.close(fig)


# ---- figure 2: VIB β sweep --------------------------------------------------------------------
def beta_sweep(g: pd.DataFrame):
    vib = g[g.method == "vib"]
    if vib.empty:
        return
    panels = [("val_acc", "MNLI accuracy  ↑"), ("hans_acc", "HANS accuracy (OOD)  ↑"),
              ("val_ece", "ECE  ↓"), ("gap_nll", "Generalization gap (NLL)  ↓")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (m, title) in zip(axes, panels, strict=True):
        for n, s in vib.groupby("n_train"):
            s = s.sort_values("beta")
            ax.plot(s.beta, s[f"{m}_mean"], "o-", color=FAMILY["vib"][1], label=f"VIB, n={n:,}")
            base = g[(g.method == "none") & (g.n_train == n)]
            if not base.empty:
                ax.axhline(base[f"{m}_mean"].iloc[0], color=AXIS, lw=1, zorder=1)
                ax.text(s.beta.min(), base[f"{m}_mean"].iloc[0],
                        "no regularization", color=MUTED, fontsize=7.5, va="bottom", ha="left")
        if m == "hans_acc":
            ax.axhline(0.5, color=AXIS, lw=1, zorder=1)
            ax.text(vib.beta.min(), 0.5, "chance", color=MUTED, fontsize=7.5, va="bottom", ha="left")
        ax.set_xscale("log"); ax.set_xlabel("β  (weight of the KL bound on I(X;Z))")
        ax.set_title(title, loc="left"); ax.grid(axis="x", visible=False)
    fig.suptitle("Variational Information Bottleneck — the compression / performance trade-off",
                 x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "vib_beta_sweep.png", dpi=170)
    plt.close(fig)

    # information kept vs generalization gap
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for _, s in vib.groupby("n_train"):
        s = s.sort_values("beta")
        axes[0].plot(s.beta, s.val_kl_mean, "o-", color=FAMILY["vib"][1])
        axes[1].plot(s.val_kl_mean, s.gap_nll_mean, "o-", color=FAMILY["vib"][1])
        for _, r in s.iterrows():
            axes[1].annotate(f"β={r.beta:g}", (r.val_kl_mean, r.gap_nll_mean), xytext=(7, 4),
                             textcoords="offset points", fontsize=8, color=INK2)
    axes[0].set_xscale("log"); axes[0].set_xlabel("β")
    axes[0].set_title("Information kept:  KL( q(z|x) ‖ N(0,I) )  in nats  ↓", loc="left")
    axes[1].set_xlabel("information kept (nats)")
    axes[1].set_title("Less information kept → smaller gap", loc="left")
    axes[1].set_ylabel("val NLL − train NLL")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "vib_information.png", dpi=170)
    plt.close(fig)


# ---- figure 3: reliability diagram -----------------------------------------------------------
def pick_curves(df: pd.DataFrame, n: int) -> list[str]:
    """one run per method family for line plots: baseline + best-ECE config of each regularizer."""
    sub = df[(df.n_train == n) & (df.seed == df.seed.min())]
    runs = []
    for m in ["none", "label_smoothing", "confidence_penalty", "vib"]:
        s = sub[sub.method == m]
        if not s.empty:
            runs.append(s.loc[s.val_ece.idxmin(), "run"])
    return runs


def reliability(df: pd.DataFrame, n: int, n_bins: int = 10, min_count: int = 25):
    runs = pick_curves(df, n)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, zorder=1)
    ax.text(0.97, 0.97, "perfect calibration", color=MUTED, fontsize=7.5, ha="right", va="top", rotation=45,
            rotation_mode="anchor", transform=ax.transData)
    for run in runs:
        m = json.loads((RES / run / "metrics.json").read_text())
        z = np.load(RES / run / "predictions_val.npz")
        p = np.exp(z["logits"] - z["logits"].max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        conf, correct = p.max(-1), (p.argmax(-1) == z["labels"]).astype(float)
        edges = np.linspace(1 / 3, 1, n_bins + 1)
        xs, ys = [], []
        for lo, hi in pairwise(edges):
            mask = (conf > lo) & (conf <= hi)
            if mask.sum() >= min_count:
                xs.append(conf[mask].mean()); ys.append(correct[mask].mean())
        r = df[df.run == run].iloc[0]
        ax.plot(xs, ys, "o-", color=CURVE[r.method], zorder=3,
                label=f"{label(r)}   ECE {m['val']['ece']:.3f}")
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability on MNLI validation  (n_train = {n:,})", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / f"reliability_n{n}.png", dpi=170)
    plt.close(fig)


# ---- figure 4: training dynamics -------------------------------------------------------------
def dynamics(df: pd.DataFrame, n: int):
    runs = pick_curves(df, n)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for run in runs:
        m = json.loads((RES / run / "metrics.json").read_text())
        h = pd.DataFrame(m["history"])
        r = df[df.run == run].iloc[0]
        axes[0].plot(h.epoch, h.train_ce, "o-", color=CURVE[r.method], label=label(r))
        axes[1].plot(h.epoch, h.val_nll, "o-", color=CURVE[r.method])
        axes[2].plot(h.epoch, h.val_ece, "o-", color=CURVE[r.method])
    for ax, t in zip(axes, ["Train cross-entropy", "Validation NLL  ↓", "Validation ECE  ↓"], strict=True):
        ax.set_title(t, loc="left"); ax.set_xlabel("epoch"); ax.grid(axis="x", visible=False)
        ax.set_xticks(sorted(h.epoch))
    axes[0].legend(loc="upper right")
    fig.suptitle("Memorization in real time: the baseline fits the train set while its validation NLL rises",
                 x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / f"training_dynamics_n{n}.png", dpi=170)
    plt.close(fig)


# ---- main -------------------------------------------------------------------------------------
def main():
    FIG.mkdir(exist_ok=True)
    df = load()
    g = agg(df)
    cols = ["run", "train_acc", "val_acc", "hans_acc", "val_ece", "gap_nll", "val_kl", "time_min"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for n in sorted(df.n_train.unique()):
        overview(g, n)
        reliability(df, n)
        dynamics(df, n)
    beta_sweep(g)
    print(f"\nwrote {RES / 'summary.csv'} and {len(list(FIG.glob('*.png')))} figures in figures/")


if __name__ == "__main__":
    main()
