"""Run a single configuration.

Examples:
  uv run scripts/run_experiment.py --method none --n_train 1000
  uv run scripts/run_experiment.py --method vib --beta 1e-3 --n_train 1000
  uv run scripts/run_experiment.py --method confidence_penalty --beta 0.1
  uv run scripts/run_experiment.py --method label_smoothing --epsilon 0.1
  uv run scripts/run_experiment.py --method weight_decay --weight_decay 0.1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inforeg.config import METHODS, Config  # noqa: E402
from inforeg.train import run  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, default="none")
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_eval", type=int, default=2000)
    p.add_argument("--n_hans", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--run_name", default="")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    cfg = Config(**{k: v for k, v in vars(a).items() if k != "force"})
    run(cfg, force=a.force)


if __name__ == "__main__":
    main()
