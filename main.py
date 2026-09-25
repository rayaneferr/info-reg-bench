"""info-reg-bench — single entry point.

  uv run main.py smoke                       # 1-minute end-to-end check of your install
  uv run main.py run --method vib --beta 1e-3
  uv run main.py run --method vib --beta 1e-3 --z_dim 32 --lora_r 8
  uv run main.py grid --n_train 1000
  uv run main.py figures
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inforeg.config import METHODS, Config

# Every Config field becomes a flag; these add choices / help on top of the dataclass defaults.
CHOICES = {"method": METHODS, "device": ("auto", "mps", "cuda", "cpu")}
HELP = {
    "beta": "strength of confidence_penalty / vib",
    "epsilon": "label smoothing",
    "weight_decay": "AdamW decoupled weight decay",
    "dropout": "dropout before the linear head",
    "n_train": "MNLI training examples (subsampled with --seed)",
    "n_eval": "MNLI validation_matched subsample",
    "n_hans": "HANS subsample (out-of-domain)",
    "z_dim": "VIB bottleneck size",
    "model_name": "any causal LM on the Hub, e.g. HuggingFaceTB/SmolLM2-135M for CPU",
    "run_name": "defaults to <method>_n<n_train>_<hyper>_s<seed>",
}
NOT_FLAGS = {"tags"}
GRID_OWNED = {"method", "beta", "epsilon", "weight_decay", "n_train", "seed", "run_name"}


def add_config_args(p: argparse.ArgumentParser, skip: set[str] = frozenset()):
    for f in dataclasses.fields(Config):
        if f.name in NOT_FLAGS | skip:
            continue
        kw = {"default": f.default, "help": HELP.get(f.name, "") + f"  (default: {f.default!r})"}
        if f.name in CHOICES:
            kw["choices"] = CHOICES[f.name]
        else:
            kw["type"] = type(f.default)
        p.add_argument(f"--{f.name}", **kw)


def config_kwargs(a: argparse.Namespace, skip: set[str] = frozenset()) -> dict:
    names = {f.name for f in dataclasses.fields(Config)} - skip
    return {k: v for k, v in vars(a).items() if k in names}


def cmd_run(a):
    from inforeg.train import run
    run(Config(**config_kwargs(a)), force=a.force)


def cmd_smoke(a):
    from inforeg.train import run
    cfg = Config(method="vib", beta=1e-3, n_train=64, n_eval=64, n_hans=64, epochs=1,
                 model_name=a.model_name, device=a.device, out_dir="results/_smoke", run_name="smoke")
    r = run(cfg, force=True)
    print(f"\nsmoke test OK — install works. val_acc={r['val']['accuracy']:.3f} (meaningless at this size)")


def cmd_grid(a):
    from inforeg.grid import grid_configs
    from inforeg.train import run
    cfgs = grid_configs(a.n_train, a.seeds, **config_kwargs(a, skip=GRID_OWNED))
    print(f"{len(cfgs)} runs:"); [print(" -", c.run_name) for c in cfgs]
    if not a.dry_run:
        for c in cfgs:
            run(c)


def cmd_figures(a):
    import runpy
    runpy.run_path(str(Path(__file__).parent / "scripts" / "make_figures.py"), run_name="__main__")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="info-reg-bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="train + evaluate one configuration")
    add_config_args(s)
    s.add_argument("--force", action="store_true", help="re-run even if metrics.json exists")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("smoke", help="tiny end-to-end run to check the install (~1 min)")
    s.add_argument("--model_name", default=Config.model_name)
    s.add_argument("--device", default=Config.device, choices=CHOICES["device"])
    s.set_defaults(func=cmd_smoke)

    s = sub.add_parser("grid", help="run the whole grid (skips finished runs); extra flags apply to all")
    s.add_argument("--n_train", type=int, nargs="+", default=[1000, 5000])
    s.add_argument("--seeds", type=int, nargs="+", default=[0])
    s.add_argument("--dry_run", action="store_true")
    add_config_args(s, skip=GRID_OWNED)
    s.set_defaults(func=cmd_grid)

    s = sub.add_parser("figures", help="aggregate results/ into summary.csv + figures/")
    s.set_defaults(func=cmd_figures)
    return p


def main(argv: list[str] | None = None):
    a = build_parser().parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
