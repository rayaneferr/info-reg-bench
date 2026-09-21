# Contributing

Issues and PRs are welcome — especially new regularizers and new evaluation axes.

## Add a regularizer

1. Add its name to `METHODS` in `inforeg/config.py`.
2. Implement the penalty in `inforeg/losses.py` (one branch, one docstring line with the formula).
   If it needs an architectural change (a bottleneck, an extra head), extend `Classifier` in `inforeg/model.py`.
3. Add a unit test in `tests/test_losses.py` that checks the closed form on a random tensor.
4. Add one entry to `GRID` in `scripts/run_grid.py` and a row to the methods table in `README.md`.

## Before opening a PR

```bash
make lint test
uv run main.py smoke
```

Please keep `results/**/metrics.json` out of PRs unless you are adding a full, documented grid.
