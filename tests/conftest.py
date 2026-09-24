"""Shared test helpers: fixture loading + optional-dependency guards.

Core tests avoid network and heavy ML stacks.  Tests that need torch or the
ModernBERT tokenizer are marked ``pytest.mark.train`` and skipped when the
train extra is not installed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # runnable without an editable install

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "sample_rows.jsonl"


def fixture_rows() -> list[dict]:
    """Rows from tests/fixtures/sample_rows.jsonl (comment lines skipped)."""
    with FIXTURE_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip() and not line.startswith("#")]


def transformers_available() -> bool:
    """True when the transformers train extra is installed (lazy import)."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


requires_transformers = pytest.mark.skipif(
    not transformers_available(),
    reason="transformers not installed (train extra) — tokenizer-dependent test",
)
