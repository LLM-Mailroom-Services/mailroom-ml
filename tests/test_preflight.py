"""Operator preflight CLI."""

import json
import sys
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _load_preflight_module():
    path = ROOT / "training" / "preflight.py"
    spec = spec_from_loader("preflight_cli", SourceFileLoader("preflight_cli", str(path)))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_preflight_synthetic_policy_ok(tmp_path: Path):
    mod = _load_preflight_module()
    report = tmp_path / "report.json"
    missing_stage = tmp_path / "no-stage"
    with patch.object(sys, "argv", [
        "preflight",
        "--stage", str(missing_stage),
        "--report", str(report),
    ]):
        code = mod.main()
    assert code == 1
    data = json.loads(report.read_text())
    assert data["checks"]["synthetic_policy"]["ok"] is True
    assert data["checks"]["stage"]["ok"] is False
