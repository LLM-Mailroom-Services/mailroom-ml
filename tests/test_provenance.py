"""Provenance primitives: manifest facts are computed from the data, sidecar
writes are atomic, content hashes never become pseudo-hashes."""
from __future__ import annotations

import os

import pytest

from conftest import fixture_rows
from mailroom_ml.dataset import build_documents, stage
from mailroom_ml.labels import label_maps
from mailroom_ml.provenance import atomic_write_text, build_manifest, clean_sha256


def _manifest(docs) -> str:
    counts = {"documents": docs["split"].value_counts().to_dict()}
    return build_manifest(docs, counts, label_maps(docs))


def test_manifest_test_count_is_computed_not_hardcoded():
    docs = build_documents(fixture_rows())
    n_test = int((docs["split"] == "test").sum())
    assert n_test != 323
    text = _manifest(docs)
    assert f"corpus test ({n_test})" in text
    assert "corpus test (323)" not in text


def test_manifest_leak_audit_line_reflects_the_data():
    docs = build_documents(fixture_rows())
    assert "title==filename 0, title==filename-stem 0, filename-shaped titles 0" in _manifest(docs)
    leaky = docs.copy()
    leaky.loc[leaky.index[0], "title"] = leaky.loc[leaky.index[0], "filename"]
    leaky.loc[leaky.index[1], "title"] = leaky.loc[leaky.index[1], "filename"].rsplit(".", 1)[0]
    text = _manifest(leaky)
    assert "title==filename 1," in text
    assert "title==filename-stem 1," in text  # only the extension-less row
    assert "leak_audit       : title==filename 0" not in text


def test_manifest_is_byte_deterministic_across_stage_rebuilds(tmp_path):
    stage(tmp_path / "a", rows=fixture_rows(), with_windows=False)
    stage(tmp_path / "b", rows=fixture_rows(), with_windows=False)
    assert (tmp_path / "a" / "manifest.txt").read_bytes() == (tmp_path / "b" / "manifest.txt").read_bytes()


@pytest.mark.parametrize("value", [None, float("nan"), "", "  ", "nan", "None", "<NA>"])
def test_clean_sha256_null_like_is_empty(value):
    assert clean_sha256(value) == ""


def test_clean_sha256_keeps_real_hashes():
    assert clean_sha256(" abc123 ") == "abc123"


def test_atomic_write_text_replaces_whole_file_or_nothing(tmp_path, monkeypatch):
    target = tmp_path / "labels.json"
    atomic_write_text(target, "v1\n")
    assert target.read_text() == "v1\n"
    atomic_write_text(target, "v2\n")
    assert target.read_text() == "v2\n"
    assert not list(tmp_path.glob(".*.tmp"))

    def _boom(src, dst):
        raise OSError("interrupted")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError, match="interrupted"):
        atomic_write_text(target, "v3-partial\n")
    assert target.read_text() == "v2\n"           # old content untouched
    assert not list(tmp_path.glob(".*.tmp"))       # tmp cleaned up
