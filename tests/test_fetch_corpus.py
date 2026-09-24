"""fetch_corpus CLI (mocked Hub download)."""

import sys
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _load_fetch_module():
    path = ROOT / "training" / "fetch_corpus.py"
    spec = spec_from_loader("fetch_corpus_cli", SourceFileLoader("fetch_corpus_cli", str(path)))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_fetch_corpus_dry_run():
    mod = _load_fetch_module()
    with patch.object(sys, "argv", ["fetch_corpus", "--dry-run"]):
        assert mod.main() == 0


def test_fetch_corpus_verifies_row_count(tmp_path):
    mod = _load_fetch_module()
    with (
        patch.object(sys, "argv", ["fetch_corpus", "--local-dir", str(tmp_path)]),
        patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)),
        patch.object(mod, "load_corpus_rows", return_value=[{}] * 100),
    ):
        assert mod.main() == 1
