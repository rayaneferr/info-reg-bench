"""Run the full V1 grid (skips runs that already have metrics.json).

  uv run scripts/run_grid.py                 # everything
  uv run scripts/run_grid.py --n_train 1000  # one training size
  uv run scripts/run_grid.py --dry_run       # just list
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inforeg.config import Config
from inforeg.train import run

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, nargs="+", default=[1000, 5000])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    cfgs = [Config(**g, n_train=n, seed=s) for n in a.n_train for s in a.seeds for g in GRID]
    print(f"{len(cfgs)} runs")
    for c in cfgs:
        print(" -", c.run_name)
    if a.dry_run:
        return
    for c in cfgs:
        run(c)


if __name__ == "__main__":
    main()
