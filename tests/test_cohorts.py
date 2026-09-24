"""Cohort builder tests (mailroom-issues #104 M7a).

Hermetic: tiny pandas frames, no network, no model.  Pins the cohort
split (short/long, single/multi-window), dup tracking via content_sha256,
the clerk `looks_messy` heuristic, and the deterministic long-doc
supplement seam.
"""
from __future__ import annotations

import pandas as pd

from mailroom_ml.cohorts import (
    build_cohort_report,
    build_supplement,
    char_lengths,
    cohort_split,
    duplicate_groups,
    looks_messy,
    stratify_counts,
    window_cohort,
)
from mailroom_ml.config import BERT_INTAKE_MAX_CHARS


def _docs():
    return pd.DataFrame([
        {"filename": "a.txt", "doc_text": "x" * 100, "doc_type": "contract",
         "subclass": "service", "content_sha256": "h1", "token_estimate": 30},
        {"filename": "b.txt", "doc_text": "y" * 100, "doc_type": "contract",
         "subclass": "service", "content_sha256": "h1", "token_estimate": 30},
        {"filename": "c.txt", "doc_text": "z" * 100, "doc_type": "contract",
         "subclass": "license", "content_sha256": "h2", "token_estimate": 30},
        {"filename": "d.txt", "doc_text": "w" * 40_000, "doc_type": "contract",
         "subclass": "service", "content_sha256": "h3", "token_estimate": 11_000},
        {"filename": "e.txt", "doc_text": "v" * 100, "doc_type": "insurance_claim",
         "subclass": "carrier", "content_sha256": "h4", "token_estimate": 30},
    ])


def test_char_lengths_and_cohort_split():
    d = cohort_split(_docs())
    assert char_lengths(_docs()).tolist() == [100, 100, 100, 40_000, 100]
    assert d["cohort"].tolist() == ["short", "short", "short", "long", "short"]
    # the long doc exceeds the context-fit gate -> routes LLM on fit alone
    long = d[d["cohort"] == "long"]
    assert len(long) == 1 and long.iloc[0]["filename"] == "d.txt"


def test_window_cohort_uses_token_estimate():
    d = window_cohort(_docs())
    assert d["window_cohort"].tolist() == [
        "single-window", "single-window", "single-window",
        "multi-window", "single-window"]


def test_window_cohort_char_fallback_without_estimate():
    d = _docs().drop(columns=["token_estimate"])
    out = window_cohort(d)
    # 40k chars / 3.8 ~ 10.5k tokens > 8192 -> multi-window
    assert out["window_cohort"].tolist() == [
        "single-window", "single-window", "single-window",
        "multi-window", "single-window"]


def test_stratify_counts():
    assert stratify_counts(_docs()) == {"contract": 4, "insurance_claim": 1}


def test_duplicate_groups_tracks_sha256_families():
    dup = duplicate_groups(_docs())
    assert dup["n_groups"] == 1
    assert dup["n_duplicate_rows"] == 2
    assert dup["groups"] == [["a.txt", "b.txt"]]
    # absent column -> no groups, never a crash
    assert duplicate_groups(_docs().drop(columns=["content_sha256"])) == {
        "n_groups": 0, "n_duplicate_rows": 0, "groups": []}


def test_duplicate_groups_skips_empty_sha256():
    docs = pd.DataFrame([
        {"filename": "a.txt", "content_sha256": ""},
        {"filename": "b.txt", "content_sha256": "nan"},
        {"filename": "c.txt", "content_sha256": "real"},
        {"filename": "d.txt", "content_sha256": "real"},
    ])
    dup = duplicate_groups(docs)
    assert dup["n_groups"] == 1
    assert dup["groups"] == [["c.txt", "d.txt"]]


def test_looks_messy_heuristic():
    assert looks_messy("clean text with words") is False
    assert looks_messy("") is True
    assert looks_messy("bad\x00control") is True
    assert looks_messy("replacement \ufffd char") is True
    assert looks_messy("!!!!!?????%%%%%") is True  # non-word dominated


def test_build_supplement_deterministic_and_long_only():
    d = _docs()
    sup = build_supplement(d, per_class=2, min_chars=1_000, seed=42)
    assert sup["coverage_only"].all()
    assert set(sup["filename"]) == {"d.txt"}  # only the 40k-char doc
    # deterministic
    sup2 = build_supplement(d, per_class=2, min_chars=1_000, seed=42)
    assert sup["filename"].tolist() == sup2["filename"].tolist()


def test_build_cohort_report_table_shape():
    rep = build_cohort_report(_docs())
    assert rep["n_docs"] == 5
    assert rep["by_doc_type"]["contract"] == {
        "short": 3, "long": 1, "single_window": 3, "multi_window": 1,
        "messy": 0}
    assert rep["by_doc_type"]["insurance_claim"] == {
        "short": 1, "long": 0, "single_window": 1, "multi_window": 0,
        "messy": 0}
    assert rep["duplicates"]["n_groups"] == 1
    assert "single-window agreement is trivially 1.0" in rep["note"]


def test_cohort_split_respects_config_gate():
    assert BERT_INTAKE_MAX_CHARS == 30_000  # the context-fit gate constant
    d = cohort_split(_docs())
    # 40k chars > 30k gate -> long; 100 chars -> short
    assert (d["cohort"] == "long").sum() == 1
