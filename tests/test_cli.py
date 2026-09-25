"""CLI and grid wiring: flags generated from Config, grid overrides, run naming."""
import dataclasses

import pytest

import main
from inforeg.config import Config
from inforeg.grid import GRID, grid_configs


def _parse(*argv):
    return main.build_parser().parse_args(list(argv))


def test_every_config_field_is_a_run_flag():
    a = _parse("run")
    fields = {f.name for f in dataclasses.fields(Config)} - main.NOT_FLAGS
    assert fields <= vars(a).keys()
    assert Config(**main.config_kwargs(a)) == Config()


def test_run_flags_reach_the_config():
    a = _parse("run", "--method", "vib", "--beta", "1e-3", "--z_dim", "32", "--lora_r", "8")
    cfg = Config(**main.config_kwargs(a))
    assert (cfg.method, cfg.beta, cfg.z_dim, cfg.lora_r) == ("vib", 1e-3, 32, 8)


def test_invalid_method_is_rejected_by_argparse():
    with pytest.raises(SystemExit):
        _parse("run", "--method", "dropout")


def test_grid_crosses_configs_sizes_and_seeds():
    cfgs = grid_configs([1000, 5000], [0, 1, 2])
    assert len(cfgs) == len(GRID) * 2 * 3
    assert len({c.run_name for c in cfgs}) == len(cfgs)


def test_grid_overrides_apply_to_every_run_and_are_tagged():
    a = _parse("grid", "--n_train", "1000", "--seeds", "0", "--epochs", "3")
    cfgs = grid_configs(a.n_train, a.seeds, **main.config_kwargs(a, skip=main.GRID_OWNED))
    assert {c.epochs for c in cfgs} == {3}
    default_names = {c.run_name for c in grid_configs([1000], [0])}
    assert not default_names & {c.run_name for c in cfgs}


def test_default_run_names_are_stable():
    """Committed results/ rely on these names — changing them silently re-runs the grid."""
    assert Config(method="none").run_name == "none_n1000_s0"
    assert Config(method="weight_decay", weight_decay=1.0).run_name == "weight_decay_n1000_wd1_s0"
    cp = Config(method="confidence_penalty", beta=0.05, seed=2)
    assert cp.run_name == "confidence_penalty_n1000_b0.05_s2"


def test_non_default_hyperparameters_are_tagged():
    assert Config(method="vib", beta=1e-3, z_dim=64, epochs=3).run_name == "vib_n1000_b0.001_s0_z64_ep3"
    assert Config(model_name="HuggingFaceTB/SmolLM2-135M").run_name == "none_n1000_s0_SmolLM2-135M"
    assert Config(device="cpu", out_dir="elsewhere").run_name == "none_n1000_s0"   # no effect on results
