"""info-reg-bench — single entry point.

  uv run main.py smoke                       # 1-minute end-to-end check of your install
  uv run main.py run --method vib --beta 1e-3
  uv run main.py grid --n_train 1000
  uv run main.py figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inforeg.config import METHODS, Config  # noqa: E402


def add_run_args(p: argparse.ArgumentParser):
    p.add_argument("--method", choices=METHODS, default="none")
    p.add_argument("--beta", type=float, default=0.0, help="strength of confidence_penalty / vib")
    p.add_argument("--epsilon", type=float, default=0.0, help="label smoothing")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_eval", type=int, default=2000)
    p.add_argument("--n_hans", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B",
                   help="any causal LM on the Hub, e.g. HuggingFaceTB/SmolLM2-135M for CPU")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--run_name", default="")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--force", action="store_true", help="re-run even if metrics.json exists")


def cmd_run(a):
    from inforeg.train import run
    cfg = Config(**{k: v for k, v in vars(a).items() if k not in ("cmd", "force", "func")})
    run(cfg, force=a.force)


def cmd_smoke(a):
    from inforeg.train import run
    cfg = Config(method="vib", beta=1e-3, n_train=64, n_eval=64, n_hans=64, epochs=1,
                 model_name=a.model_name, device=a.device, out_dir="results/_smoke", run_name="smoke")
    r = run(cfg, force=True)
    print("\nsmoke test OK — install works. val_acc=%.3f (meaningless at this size)" % r["val"]["accuracy"])


def cmd_grid(a):
    from inforeg.train import run
    from scripts.run_grid import GRID
    cfgs = [Config(**g, n_train=n, seed=s, device=a.device, model_name=a.model_name)
            for n in a.n_train for s in a.seeds for g in GRID]
    print(f"{len(cfgs)} runs:"); [print(" -", c.run_name) for c in cfgs]
    if not a.dry_run:
        for c in cfgs:
            run(c)


def cmd_figures(a):
    import runpy
    runpy.run_path(str(Path(__file__).parent / "scripts" / "make_figures.py"), run_name="__main__")


def main():
    p = argparse.ArgumentParser(prog="info-reg-bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="train + evaluate one configuration")
    add_run_args(s); s.set_defaults(func=cmd_run)
    s = sub.add_parser("smoke", help="tiny end-to-end run to check the install (~1 min)")
    s.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    s.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    s.set_defaults(func=cmd_smoke)
    s = sub.add_parser("grid", help="run the whole V1 grid (skips finished runs)")
    s.add_argument("--n_train", type=int, nargs="+", default=[1000, 5000])
    s.add_argument("--seeds", type=int, nargs="+", default=[0])
    s.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    s.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    s.add_argument("--dry_run", action="store_true"); s.set_defaults(func=cmd_grid)
    s = sub.add_parser("figures", help="aggregate results/ into summary.csv + figures/")
    s.set_defaults(func=cmd_figures)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
