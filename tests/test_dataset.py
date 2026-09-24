"""Dataset pipeline tests (port of corpus-eda test_prep.py + leak-control).

Runs against the committed synthetic fixture rows (no snapshot needed) plus
the new grouped-split / sha-dedup / leakage-audit primitives.  The
full-corpus contract test is marked ``fullcorpus`` and skipped when the
local snapshot under ``data/parquet`` is absent (gitignored — fetched via
the build CLI / snapshot download).
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from conftest import fixture_rows, requires_transformers, transformers_available
from mailroom_ml.config import DATA_DIR, DOC_TYPES
from mailroom_ml.dataset import (
    build_documents,
    build_windows,
    dedup_by_sha,
    grouped_split,
    has_adopted_enrichment,
    leakage_audit,
    load_corpus_rows,
    refresh_dataset_info,
    stage,
    stage_stats,
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
    # title-wins: subject beats filename, exhibit_description second, and a
    # row with no semantic title gets "" (the filename fallback was removed
    # in the pre-flight clean — it leaked the label)
    assert docs[docs["filename"] == "enron_subject_001.txt"].iloc[0]["title"] == "Re: Enron"
    assert docs[docs["filename"] == "edgar_exhibit_001.htm"].iloc[0]["title"] == "Certificates of Officer"
    assert docs[docs["filename"] == "enron_notice_001.txt"].iloc[0]["title"] == ""
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
@pytest.mark.train
def test_build_windows_fixtures():
    docs = build_documents(fixture_rows())
    wins = build_windows(docs)
    # fixtures are tiny: the 10% val draw can be empty — but test is ALWAYS out
    assert set(wins["split"]) <= {"train", "validation"}
    assert "test" not in set(wins["split"])
    assert (wins["window_index"] < wins["n_windows"]).all()
    assert wins["filename"].nunique() == (docs["split"] != "test").sum()


@requires_transformers
@pytest.mark.train
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
    assert not (tmp_path / "data" / "windows").exists()
    check = verify_stage(tmp_path)
    assert check["ok"], check["problems"]


def test_stage_stats_sums_all_files_and_refresh_info(tmp_path):
    """The enrichment parquet lives BESIDE the canonical one; the true train
    count is the sum, and dataset_info.json must advertise it after
    refresh (stage() writes it canonical-only)."""
    stage(tmp_path, rows=fixture_rows(), with_windows=False)
    assert not has_adopted_enrichment(tmp_path)
    n_train = stage_stats(tmp_path)["counts"]["documents"]["train"]

    d = tmp_path / "data" / "documents" / "train"
    extra = pd.read_parquet(d / "train-00000-of-00001.parquet").head(2).copy()
    extra["filename"] = ["zzzz-extra-0", "zzzz-extra-1"]
    extra["split"] = "train"
    extra.to_parquet(d / "enrichment-00000-of-00001.parquet")

    assert has_adopted_enrichment(tmp_path)
    assert stage_stats(tmp_path)["counts"]["documents"]["train"] == n_train + 2
    # verify_stage still passes over the enriched tree (rows checks all files)
    assert verify_stage(tmp_path)["rows"] == len(fixture_rows()) + 2

    refresh_dataset_info(tmp_path)
    info = json.loads((tmp_path / "dataset_info.json").read_text())
    assert info["documents"]["splits"]["train"]["num_examples"] == n_train + 2


@requires_transformers
@pytest.mark.train
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
        af = sorted((a_dir / "data" / cfg_split).glob("*.parquet"))
        bf = sorted((b_dir / "data" / cfg_split).glob("*.parquet"))
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
    not (DATA_DIR / "data" / "ground_truth" / "train").exists(),
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
    or not (DATA_DIR / "data" / "ground_truth" / "train").exists(),
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


# ---------------------------------------------------------------------------
# Pre-flight clean (2026-09-20): intake clerk + label-leak gate
# ---------------------------------------------------------------------------

def test_deterministic_normalize_is_the_pipeline_clerk():
    """The vendored clerk removes BOM/CRLF/blank-runs/controls — the exact
    skew between raw corpus text and what the pipeline feeds the classifier."""
    from mailroom_ml.normalize import deterministic_normalize

    raw = "\ufeffAGREEMENT  \r\n\r\n\r\n\r\nby and among  \r\n\r\n  X   Y  \r\n"
    cleaned, stats = deterministic_normalize(raw)
    assert "\ufeff" not in cleaned and "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned == "AGREEMENT\n\nby and among\n\nX Y"
    assert stats["changed"] is True
    # empty input is a no-op, never a crash
    assert deterministic_normalize("") == ("", {
        "raw_chars": 0, "cleaned_chars": 0, "collapsed_blank_runs": 0,
        "hyphen_unwraps": 0, "changed": False})


def test_build_title_is_semantic_only():
    """No filename fallback: a row with no subject/exhibit gets '' (leak fix)."""
    from mailroom_ml.preprocessing import build_title

    assert build_title({"filename": "auto:CLM-1.txt", "metadata": {}}) == ""
    assert build_title({"filename": "x.txt",
                        "metadata": {"subject": "Re: Enron"}}) == "Re: Enron"
    assert build_title({"filename": "x.htm", "metadata": {
        "exhibit_description": "Certificates of Officer"}}) == "Certificates of Officer"


def test_filename_leak_audit_flags_and_clears():
    from mailroom_ml.dataset import filename_leak_audit

    leaky = pd.DataFrame({
        "filename": ["auto:CLM-1.txt", "contract_1_merger_agreement.txt"],
        "title": ["auto:CLM-1.txt", "contract_1_merger_agreement.txt"],
        "doc_text": ["body", "body"],
        "doc_type": ["insurance_claim", "merger_agreement"],
        "subclass": ["auto", "all_stock"],
    })
    audit = filename_leak_audit(leaky)
    assert audit["clean"] is False
    assert audit["title_eq_filename"] == 2
    assert audit["title_looks_like_filename"] == 2

    clean = leaky.assign(title=["", ""])
    audit2 = filename_leak_audit(clean)
    assert audit2["clean"] is True
    assert audit2["title_eq_filename"] == 0
    assert audit2["title_looks_like_filename"] == 0


def test_verify_stage_fails_loudly_on_filename_leak(tmp_path):
    """A stage whose titles are filenames must FAIL verify (the gate)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    docs = pd.DataFrame({
        "filename": ["a.txt"], "document_id": [""], "content_sha256": [""],
        "source_revision": ["r"], "title": ["a.txt"], "doc_text": ["body"],
        "doc_type": ["contract"], "subclass": ["consulting"],
        "corpus_split": ["train"], "token_estimate": [1], "split": ["train"],
    })
    for split in ("train", "validation", "test"):
        d = tmp_path / "data" / "documents" / split
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(docs, preserve_index=False),
                       d / f"{split}-00000-of-00001.parquet")
    chk = verify_stage(tmp_path)
    assert chk["ok"] is False
    assert any("label leak" in p for p in chk["problems"])
