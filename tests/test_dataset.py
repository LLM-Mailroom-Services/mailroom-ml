"""Dataset pipeline tests (port of corpus-eda test_prep.py + leak-control).

Runs against the committed synthetic fixture rows (no snapshot needed) plus
the new grouped-split / sha-dedup / leakage-audit primitives.  The
full-corpus contract test is marked ``fullcorpus`` and skipped when the
local snapshot under ``data/parquet`` is absent (gitignored — fetched via
the build CLI / snapshot download).
"""
from __future__ import annotations

import pandas as pd
import pytest

from conftest import fixture_rows, requires_transformers, transformers_available
from mailroom_ml.config import DATA_DIR, DOC_TYPES
from mailroom_ml.dataset import (
    build_documents,
    build_windows,
    dedup_by_sha,
    grouped_split,
    leakage_audit,
    load_corpus_rows,
    stage,
    stratified_split,
    verify_stage,
)
from mailroom_ml.labels import SUBCLASS_BY_CLASS

EXPECTED_COLUMNS = [
    "filename", "document_id", "content_sha256", "source_revision", "title",
    "doc_text", "doc_type", "subclass", "corpus_split", "token_estimate",
    "split",
]


def test_stratified_split_deterministic_and_stratified():
    df = pd.DataFrame({
        "filename": [f"f{i:04d}" for i in range(100)],
        "doc_type": ["contract"] * 90 + ["merger_agreement"] * 10,
    })
    a = stratified_split(df)
    b = stratified_split(df)
    assert a.equals(b), "split must be byte-deterministic"
    assert set(a["split"]) == {"train", "validation"}
    for cls, grp in a.groupby("doc_type"):
        n_val = grp["split"].eq("validation").sum()
        assert n_val == round(len(grp) * 0.1), cls
    # no filename in two splits
    assert not a["filename"].duplicated().any()


def test_build_documents_fixtures():
    rows = fixture_rows()
    docs = build_documents(rows)
    assert len(docs) == len(rows)
    assert list(docs.columns) == EXPECTED_COLUMNS
    assert set(docs["doc_type"]) <= set(DOC_TYPES)
    # fixture rows: 1 test (cms_outpatient_001) — held out as test
    assert set(docs["split"]) <= {"train", "validation", "test"}
    assert (docs["split"] == "test").sum() == 1
    # canonical subclass keys only
    for cls, grp in docs.groupby("doc_type"):
        assert set(grp["subclass"]) <= set(SUBCLASS_BY_CLASS[cls]), cls
    # title-wins: subject beats filename, exhibit_description second
    assert docs[docs["filename"] == "enron_subject_001.txt"].iloc[0]["title"] == "Re: Enron"
    assert docs[docs["filename"] == "edgar_exhibit_001.htm"].iloc[0]["title"] == "Certificates of Officer"
    assert docs[docs["filename"] == "enron_notice_001.txt"].iloc[0]["title"] == "enron_notice_001.txt"
    # inject a subject row to pin the rule independent of the fixture
    injected = rows + [{
        "filename": "enron_subject_002.txt", "expected": "correspondence",
        "expected_subclass": "notice", "doc_text": "body",
        "metadata": {"subject": "Re: Enron", "original_file": ""},
        "gt_fields": {}, "prompt": "", "split": "train",
    }]
    docs2 = build_documents(injected)
    enron = docs2[docs2["filename"] == "enron_subject_002.txt"].iloc[0]
    assert enron["title"] == "Re: Enron"
    # case-folded subclass normalization (All Cash -> all_cash)
    merged = docs2[docs2["filename"] == "maud_cash_002.txt"].iloc[0]
    assert merged["subclass"] == "all_cash"


@requires_transformers
def test_build_windows_fixtures():
    docs = build_documents(fixture_rows())
    wins = build_windows(docs)
    # fixtures are tiny: the 10% val draw can be empty — but test is ALWAYS out
    assert set(wins["split"]) <= {"train", "validation"}
    assert "test" not in set(wins["split"])
    assert (wins["window_index"] < wins["n_windows"]).all()
    assert wins["filename"].nunique() == (docs["split"] != "test").sum()


@requires_transformers
def test_stage_and_verify_stage(tmp_path):
    stats = stage(tmp_path, rows=fixture_rows(), with_windows=True)
    check = verify_stage(tmp_path)
    assert check["ok"], check["problems"]
    assert check["rows"] == len(fixture_rows())
    assert stats["counts"]["documents"]["test"] == 1
    assert check["counts"]["documents"] == stats["counts"]["documents"]
    assert not check["problems"]


def test_stage_documents_only(tmp_path):
    # --no-windows path: documents config only, verify still green
    stats = stage(tmp_path, rows=fixture_rows(), with_windows=False)
    assert "windows" not in stats["counts"]
    assert not (tmp_path / "parquet" / "windows").exists()
    check = verify_stage(tmp_path)
    assert check["ok"], check["problems"]


@requires_transformers
def test_stage_byte_deterministic_rebuilds(tmp_path):
    """Rebuilds are byte-identical: manifest + sidecars + parquet bytes."""
    rows = fixture_rows()
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    stage(a_dir, rows=rows, with_windows=True)
    stage(b_dir, rows=rows, with_windows=True)
    for rel in ("manifest.txt", "labels.json", "vocabularies.json"):
        assert (a_dir / rel).read_bytes() == (b_dir / rel).read_bytes(), rel
    for cfg_split in (
        "documents/train", "documents/validation", "documents/test",
        "windows/train", "windows/validation",
    ):
        af = sorted((a_dir / "parquet" / cfg_split).glob("*.parquet"))
        bf = sorted((b_dir / "parquet" / cfg_split).glob("*.parquet"))
        assert [f.name for f in af] == [f.name for f in bf]
        assert af and all(af[i].read_bytes() == bf[i].read_bytes()
                          for i in range(len(af))), cfg_split
    # manifest carries no timestamps
    assert b"2026" not in (a_dir / "manifest.txt").read_bytes()


def test_grouped_split_families_never_straddle():
    n = 40
    df = pd.DataFrame({
        "filename": [f"f{i:03d}" for i in range(n)],
        "family": [f"fam{j // 4}" for j in range(n)],  # 10 families of 4
        "doc_type": ["contract"] * n,
    })
    a = grouped_split(df, "family", 0.2, seed=42)
    b = grouped_split(df, "family", 0.2, seed=42)
    assert a.equals(b), "grouped split must be deterministic"
    n_val_fams = int(round(10 * 0.2))
    fams_val = set(a[a["split"] == "validation"]["family"])
    fams_train = set(a[a["split"] == "train"]["family"])
    assert len(fams_val) == n_val_fams == 2
    assert fams_train.isdisjoint(fams_val)
    # every family sits entire inside one split (no straddling)
    assert set(a.groupby("family")["split"].nunique()) == {1}
    assert a["filename"].is_monotonic_increasing


def test_dedup_by_sha_filename_set_guard():
    docs = build_documents(fixture_rows())
    dup_pair = docs[
        docs["filename"].isin(["cuad_dup_a.txt", "cuad_dup_b.txt"])].copy()
    assert len(dup_pair) == 2
    # pool holds cuad_dup_a only: b (same sha, different filename) is dropped
    kept = dedup_by_sha(dup_pair, dup_pair.iloc[[0]])
    assert set(kept["filename"]) == {"cuad_dup_a.txt"}
    # same filename + same sha in pool -> same document identity, kept
    assert len(dedup_by_sha(dup_pair, dup_pair)) == 2
    # rows without a sha are opaque -> never dropped
    no_sha = docs[~docs["filename"].isin(["cuad_dup_a.txt", "cuad_dup_b.txt"])].copy()
    no_sha = no_sha.assign(content_sha256="")
    assert len(dedup_by_sha(no_sha, dup_pair)) == len(no_sha)
    # unrelated sha survives a populated pool
    other = docs[docs["filename"] == "cuad_consulting_001.txt"].copy()
    assert len(dedup_by_sha(other, dup_pair)) == 1


def test_leakage_audit_detects_cross_split_and_titles():
    df = pd.DataFrame({
        "filename": ["a.txt", "b.txt", "a.txt", "c.txt", "c.txt"],
        "title": ["Same Title", "Same Title", "Other", "X", "X"],
        "doc_type": ["contract"] * 5,
        "subclass": ["affiliate"] * 5,
        "split": ["train", "validation", "test", "train", "validation"],
    })
    audit = leakage_audit(df)
    # a/c appear in two splits each
    assert audit["duplicate_filenames_across_splits"] == {
        "a.txt": ["test", "train"], "c.txt": ["train", "validation"]}
    # two exact title groups; folded sees the same groups here
    assert audit["title_duplicates"]["exact_groups"] == 2
    assert audit["title_duplicates"]["folded_groups"] == 2
    # per-class val share: 2 validation of 4 active rows
    assert audit["per_class_val_share"] == {"contract": 0.5}
    # deterministic across calls
    assert leakage_audit(df) == audit


def test_leakage_audit_fixture_clean_and_dup_titles():
    docs = build_documents(fixture_rows())
    audit = leakage_audit(docs)
    assert audit["duplicate_filenames_across_splits"] == {}
    # fixture carries an exact duplicate title pair (both subjects "Re: Enron")
    assert audit["title_duplicates"]["exact_groups"] == 1
    assert audit["title_duplicates"]["folded_groups"] == 1
    ex = audit["title_duplicates"]["examples"]
    assert ex and ex[0]["n_rows"] == 2
    assert sorted(ex[0]["filenames"]) == ["enron_letter_002.txt", "enron_subject_001.txt"]


@pytest.mark.fullcorpus
@pytest.mark.skipif(
    not (DATA_DIR / "parquet" / "ground_truth" / "train").exists(),
    reason="local snapshot absent (data/parquet) — fetch via training/build_dataset.py",
)
def test_full_corpus_contract():
    """Full-corpus invariants: 3,302 rows, splits, no leakage, vocab."""
    rows = load_corpus_rows()
    assert len(rows) == 3302
    docs = build_documents(rows)
    # val = round(10% of each class's corpus-train count), banker's rounding
    tr = docs[docs["corpus_split"] == "train"]
    expected_val = {
        cls: int(round(n * 0.1)) for cls, n in tr.groupby("doc_type").size().items()}
    assert docs["split"].value_counts().to_dict() == {
        "train": len(tr) - sum(expected_val.values()),
        "validation": sum(expected_val.values()), "test": 323}
    assert docs[docs["split"] == "validation"].groupby("doc_type").size().to_dict() \
        == expected_val
    assert not docs["filename"].duplicated().any()
    for cls, grp in docs.groupby("doc_type"):
        assert set(grp["subclass"]) <= set(SUBCLASS_BY_CLASS[cls]), cls
    # every corpus subclass surface resolves to a canonical key (no 'other'
    # inflation beyond the corpus's own other-bucket rows)
    assert (docs["subclass"] == "other").sum() == (
        (docs["doc_type"] == "corporate_record") & (docs["subclass"] == "other")).sum() + \
        ((docs["doc_type"] == "merger_agreement") & (docs["subclass"] == "other")).sum()


@pytest.mark.fullcorpus
@pytest.mark.skipif(
    not transformers_available()
    or not (DATA_DIR / "parquet" / "ground_truth" / "train").exists(),
    reason="needs transformers (train extra) AND the local snapshot under data/parquet",
)
def test_full_corpus_windows_within_budget():
    """Every published window re-tokenizes within the model budget, title first."""
    from transformers import AutoTokenizer

    from mailroom_ml.config import MAX_TOKENS

    docs = build_documents(load_corpus_rows())
    wins = build_windows(docs)
    per_doc = wins.groupby("filename").size()
    assert len(per_doc) == 2649 + 330
    assert "test" not in set(wins["split"])
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    over = wins["text"].apply(
        lambda t: len(tok(t)["input_ids"]) > MAX_TOKENS)
    assert not over.any(), f"{int(over.sum())} windows exceed {MAX_TOKENS} tokens"
    # title prefix survives the clamp verbatim (QA 5b regression)
    bad_prefix = wins.apply(
        lambda r: not r["text"].startswith(
            docs[docs["filename"] == r["filename"]].iloc[0]["title"]), axis=1)
    assert not bad_prefix.any(), f"{int(bad_prefix.sum())} windows lost the title prefix"
