# Changelog

All notable changes to this project. The format follows [Keep a Changelog](https://keepachangelog.com),
versions follow [SemVer](https://semver.org), commits follow [Conventional Commits](https://www.conventionalcommits.org).

## [0.2.0] — 2026-09-25

### ⚠️ Breaking
- `scripts/run_grid.py` and `scripts/run_experiment.py` are removed: use `main.py grid` / `main.py run`,
  same flags (#6).

### Added
- Every `Config` field is a CLI flag (`--lora_r`, `--z_dim`, `--max_length`, `--warmup_ratio`…); `grid`
  forwards them to every run (#6).
- Non-default hyperparameters are tagged in run names (`vib_n1000_b0.001_s0_z64_ep3`) so variants never
  collide with the default runs; default names are unchanged (#6).
- `results/summary_by_config.csv` and `results/summary.md` (mean ± std over seeds); error bars on the VIB
  β-sweep figures (#10).
- Tests for the model heads on a tiny random backbone and for the CLI / run naming: 9 → 23 tests (#9).
- PR template, issue forms, Conventional-Commit PR title check, Dependabot, pre-commit hooks (#2, #5).

### Fixed
- CI and Docker installed the CUDA build of torch: `UV_TORCH_BACKEND` is ignored by `uv sync` (#1).
- `main.py figures` crashed on a fresh clone because it needed the git-ignored logits (#8).
- Validation KL was averaged over batches instead of examples (bias < 1% on the published grid) (#7).

### Changed
- Removed unused dependencies: `accelerate`, `scikit-learn`, `pyyaml`, `jupyter`, `ipykernel` (#4).
- Stricter lint: bugbear, pyupgrade, comprehensions, simplify, numpy and pytest rules (#5).

## [0.1.0] — 2026-09-21

- First release: 5 regularizers (9 configurations), Qwen2.5-0.5B + LoRA on MNLI (n_train = 1 000),
  evaluated in-domain and on HANS for accuracy, ECE, generalization gap and the VIB KL bound.

[0.2.0]: https://github.com/rayaneferr/info-reg-bench/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rayaneferr/info-reg-bench/releases/tag/v0.1.0
