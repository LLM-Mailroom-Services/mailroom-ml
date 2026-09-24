"""Synthetic policy YAML ↔ config.py interlock."""

from pathlib import Path

import pytest

from mailroom_ml.synthetic_policy import (
    assert_policy_matches_config,
    default_enrichment_caps,
    load_synthetic_policy,
    policy_version,
)


def test_load_policy_matches_config():
    policy = load_synthetic_policy()
    assert_policy_matches_config(policy)
    assert policy_version(policy) == "v1"


def test_default_enrichment_caps_keys():
    caps = default_enrichment_caps()
    assert caps["global_share"] == 0.40
    assert caps["enron_cap_mult"] == 2.0
    assert "synthetic_weight" in caps


def test_policy_drift_detected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: v1\nglobal_share: 0.99\n", encoding="utf-8")
    policy = load_synthetic_policy(bad)
    with pytest.raises(ValueError, match="global_share"):
        assert_policy_matches_config(policy)
