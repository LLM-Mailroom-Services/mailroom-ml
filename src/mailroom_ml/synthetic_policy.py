"""Load ``configs/synthetic_policy_v1.yaml`` and verify parity with ``config.py``.

The YAML is the human-readable policy artifact; ``config.SYNTHETIC_*`` remains
the import-time interlock for library code.  Enrichment loads caps from here so
operators editing the YAML see failures when constants drift.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mailroom_ml.config import (
    ROOT,
    SYNTHETIC_ELIGIBILITY_MAX_AUTHENTIC,
    SYNTHETIC_EXAMPLE_WEIGHT,
    SYNTHETIC_MAX_GLOBAL_SHARE,
    SYNTHETIC_MAX_PER_SUBCLASS_SHARE,
    SYNTHETIC_TIER_CAPS,
)

DEFAULT_POLICY_PATH = ROOT / "configs" / "synthetic_policy_v1.yaml"

_POLICY_FLOAT_KEYS = (
    ("global_share", SYNTHETIC_MAX_GLOBAL_SHARE),
    ("per_subclass_share", SYNTHETIC_MAX_PER_SUBCLASS_SHARE),
    ("example_weight", SYNTHETIC_EXAMPLE_WEIGHT),
)
_POLICY_INT_KEYS = (("eligibility_max_authentic", SYNTHETIC_ELIGIBILITY_MAX_AUTHENTIC),)


def load_synthetic_policy(path: Path | None = None) -> dict[str, Any]:
    """Parse the synthetic policy YAML (default: ``configs/synthetic_policy_v1.yaml``)."""
    policy_path = path or DEFAULT_POLICY_PATH
    if not policy_path.is_file():
        raise FileNotFoundError(f"synthetic policy not found: {policy_path}")
    raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{policy_path}: expected a YAML mapping at top level")
    return raw


def assert_policy_matches_config(policy: dict[str, Any]) -> None:
    """Fail loud when YAML numeric fields diverge from ``config.py``."""
    for key, expected in _POLICY_FLOAT_KEYS:
        got = policy.get(key)
        if got is None:
            raise ValueError(f"synthetic policy missing key {key!r}")
        if float(got) != float(expected):
            raise ValueError(
                f"synthetic policy {key}={got!r} != config {expected!r} — "
                "sync configs/synthetic_policy_v1.yaml with config.py"
            )
    for key, expected in _POLICY_INT_KEYS:
        got = policy.get(key)
        if got is None:
            raise ValueError(f"synthetic policy missing key {key!r}")
        if int(got) != int(expected):
            raise ValueError(
                f"synthetic policy {key}={got!r} != config {expected!r}"
            )
    tier_caps = policy.get("tier_caps")
    if tier_caps != [list(pair) for pair in SYNTHETIC_TIER_CAPS]:
        raise ValueError(
            "synthetic policy tier_caps diverges from config.SYNTHETIC_TIER_CAPS"
        )


def default_enrichment_caps() -> dict[str, float]:
    """Caps dict for ``assemble_enrichment`` (plan §6.2 / §6.5 + pool mults)."""
    policy = load_synthetic_policy()
    assert_policy_matches_config(policy)
    return {
        "enron_cap_mult": 2.0,
        "cuad_cap_mult": 2.0,
        "insurance_cap_mult": 2.0,
        "pseudo_max_fraction": 0.30,
        "global_share": float(policy["global_share"]),
        "per_subclass_share": float(policy["per_subclass_share"]),
        "synthetic_weight": float(policy["example_weight"]),
    }


def policy_version(policy: dict[str, Any] | None = None) -> str:
    p = policy if policy is not None else load_synthetic_policy()
    return str(p.get("version", "unknown"))
