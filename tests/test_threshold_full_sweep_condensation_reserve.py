"""``LCM_THRESHOLD_FULL_SWEEP_CONDENSATION_RESERVE_PASSES`` / ``lcm.threshold_full_sweep_condensation_reserve_passes``
(T176): passes of the threshold sweep's budget kept for summary-on-summary condensation when the leaf work hits the
pass budget. Env wins over config.yaml; a value outside [0, 16] is ignored with a warning and the next source applies.
``0`` (the default) keeps today's sweep."""

import pytest

import hermes_lcm.config as config
from hermes_lcm.config import LCMConfig

FIELD = "threshold_full_sweep_condensation_reserve_passes"
ENV = "LCM_THRESHOLD_FULL_SWEEP_CONDENSATION_RESERVE_PASSES"
YAML_SOURCE = f"config_yaml:lcm.{FIELD}"


@pytest.fixture
def yaml_lcm(monkeypatch):
    """Pin config.yaml to an ``lcm`` section the test sets (empty by default) and clear the env key."""
    section: dict = {}
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(config, "_load_hermes_config_yaml", lambda *args: {"lcm": section})
    return section


def _warned(c: LCMConfig) -> bool:
    return any(FIELD in w or ENV in w for w in c.config_source_warnings)


# -- config ---------------------------------------------------------------------------------------------------------


def test_unset_keeps_the_default_of_0_with_no_warning(yaml_lcm):
    c = LCMConfig.from_env()
    assert LCMConfig().threshold_full_sweep_condensation_reserve_passes == 0
    assert c.threshold_full_sweep_condensation_reserve_passes == 0
    assert c.config_sources[FIELD] == "default"
    assert not _warned(c)


@pytest.mark.parametrize("raw, value", [("0", 0), ("2", 2), (" 16 ", 16)])
def test_an_env_value_in_range_is_used(yaml_lcm, monkeypatch, raw, value):
    monkeypatch.setenv(ENV, raw)
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_condensation_reserve_passes == value
    assert c.config_sources[FIELD] == f"env:{ENV}"
    assert not _warned(c)


@pytest.mark.parametrize("raw", [3, "3"])
def test_a_config_yaml_value_in_range_is_used_and_not_reported_as_ignored(yaml_lcm, raw):
    yaml_lcm[FIELD] = raw
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_condensation_reserve_passes == 3
    assert c.config_sources[FIELD] == YAML_SOURCE
    assert FIELD not in c.ignored_config_yaml_lcm_keys
    assert not _warned(c)


def test_the_env_wins_over_config_yaml(yaml_lcm, monkeypatch):
    yaml_lcm[FIELD] = 3
    monkeypatch.setenv(ENV, "2")
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_condensation_reserve_passes == 2
    assert c.config_sources[FIELD] == f"env:{ENV}"


@pytest.mark.parametrize("raw", ["17", "-1", "abc", "2.5", ""])
def test_an_env_value_out_of_range_is_rejected_and_0_kept(yaml_lcm, monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_condensation_reserve_passes == 0
    assert c.config_sources[FIELD] == "default"
    assert _warned(c)


@pytest.mark.parametrize("raw", [17, -1, True, "many", 2.5])
def test_a_config_yaml_value_out_of_range_is_rejected_and_0_kept(yaml_lcm, raw):
    yaml_lcm[FIELD] = raw
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_condensation_reserve_passes == 0
    assert c.config_sources[FIELD] == "default"
    assert _warned(c)


def test_an_invalid_env_value_falls_back_to_a_valid_config_yaml_value(yaml_lcm, monkeypatch):
    yaml_lcm[FIELD] = 3
    monkeypatch.setenv(ENV, "99")
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_condensation_reserve_passes == 3
    assert c.config_sources[FIELD] == YAML_SOURCE
    assert _warned(c)
