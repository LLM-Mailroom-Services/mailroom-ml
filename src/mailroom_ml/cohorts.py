"""Cohort builder for the eval harness (mailroom-issues #104 M7a).

The published eval's cohort design hid real failure modes: ~86% of eligible
docs are single-window (agreement trivially 1.0 — the gate reduces to one
miscalibrated p), merger_agreement contributes 0 short docs, and ~7
near-dup cases exist via ``content_sha256``.  This module makes the cohort
composition explicit and reproducible:

- ``cohort_split``: short (<= ``BERT_INTAKE_MAX_CHARS`` — the context-fit
  gate) vs long docs;
- ``window_cohort``: single-window vs multi-window from the staged
  ``token_estimate`` column (char-based fallback);
- ``duplicate_groups``: near-dup tracking via ``content_sha256`` (group_id
  discipline — a dup family must not double-count in cohort stats);
- ``looks_messy``: replication of the clerk's "clean" filter heuristic;
- ``build_supplement``: deterministic per-class long-doc supplement
  (coverage-only — never accuracy) so contract/merger long-doc coverage
  exists in the report.

All functions are pure (DataFrame in -> DataFrame/dict out), no network,
no model — unit-testable with tiny frames.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

from mailroom_ml.config import BERT_INTAKE_MAX_CHARS, MAX_TOKENS, RANDOM_STATE

__all__ = [
    "char_lengths",
    "cohort_split",
    "window_cohort",
    "stratify_counts",
    "duplicate_groups",
    "looks_messy",
    "build_supplement",
    "build_cohort_report",
]

_MESSY_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MESSY_REPLACEMENT = "\ufffd"


def char_lengths(docs) -> np.ndarray:
    """Per-row ``doc_text`` character length (the context-fit gate unit)."""
    return np.asarray([len(str(t or "")) for t in docs["doc_text"]],
                      dtype=np.int64)


def cohort_split(docs, max_chars: int = BERT_INTAKE_MAX_CHARS):
    """Adds ``cohort``: "short" (fits the BERT context-fit gate) / "long".

    Short docs are the fast-path-eligible population; long docs route LLM
    on context fit alone (epic #85) — the two must never be conflated in
    coverage numbers.
    """
    out = docs.copy()
    out["cohort"] = np.where(char_lengths(out) <= max_chars, "short", "long")
    return out


def window_cohort(docs, max_tokens: int = MAX_TOKENS):
    """Adds ``window_cohort``: "single-window" / "multi-window".

    Uses the staged ``token_estimate`` column when present (the windower's
    own estimate); falls back to a char heuristic (~3.8 chars/token, the
    config's context-fit ratio).  Single-window docs have trivially 1.0
    agreement — the gate reduces to one probability, so they are scored
    separately from multi-window docs (#104).
    """
    out = docs.copy()
    if "token_estimate" in out.columns:
        est = out["token_estimate"].fillna(0).astype(np.int64).to_numpy()
    else:
        est = np.ceil(char_lengths(out) / 3.8).astype(np.int64)
    out["window_cohort"] = np.where(est <= max_tokens,
                                    "single-window", "multi-window")
    return out


def stratify_counts(docs) -> dict[str, int]:
    """Per-doc_type row counts (the stratum sizes for sampling)."""
    return dict(Counter(docs["doc_type"].astype(str)))


def duplicate_groups(docs) -> dict:
    """Near-dup families via ``content_sha256`` (group_id discipline).

    Returns ``{"n_groups": ..., "n_duplicate_rows": ..., "groups": [...]}``
    where each group lists its filenames.  A dup family must count once in
    cohort stats — the report consumer decides which member is canonical.
    """
    if "content_sha256" not in docs.columns:
        return {"n_groups": 0, "n_duplicate_rows": 0, "groups": []}
    by_hash: dict[str, list[str]] = defaultdict(list)
    for fn, h in zip(docs["filename"].astype(str),
                     docs["content_sha256"].astype(str), strict=True):
        by_hash[h].append(fn)
    groups = [sorted(fns) for fns in by_hash.values() if len(fns) > 1]
    return {
        "n_groups": len(groups),
        "n_duplicate_rows": sum(len(g) for g in groups),
        "groups": groups,
    }


def looks_messy(text: str) -> bool:
    """Clerk ``looks_messy`` heuristic replication (#104 cohort "clean" filter).

    A doc is messy when it carries control characters (excluding newlines/
    tabs), the Unicode replacement char (OCR/encoding damage), or a
    non-word-character ratio above 0.5 (scan artifacts, binary junk).
    Heuristic by design — the exact clerk rule lives in llm-mailroom; this
    is the eval-side stand-in, documented as such.
    """
    t = str(text or "")
    if not t:
        return True  # empty body is not a clean document
    if _MESSY_CONTROL.search(t) or _MESSY_REPLACEMENT in t:
        return True
    words = re.findall(r"\w", t)
    return len(words) / len(t) < 0.5


def build_supplement(docs, per_class: int, min_chars: int,
                     seed: int = RANDOM_STATE):
    """Deterministic per-class long-doc supplement (coverage-only).

    Selects up to ``per_class`` docs per doc_type with ``len(doc_text) >
    min_chars``, seeded (frozen RandomState — no sklearn).  The supplement
    exists so contract/merger long-doc COVERAGE appears in the cohort
    report; it is never accuracy data (those docs are not held-out test
    rows) — callers must mark it ``coverage_only``.
    """
    rng = np.random.RandomState(seed)
    lens = char_lengths(docs)
    long_idx = np.where(lens > min_chars)[0]
    by_class: dict[str, list[int]] = defaultdict(list)
    for i in long_idx:
        by_class[str(docs.iloc[i]["doc_type"])].append(int(i))
    picked: list[int] = []
    for cls in sorted(by_class):
        idx = sorted(by_class[cls])
        rng.shuffle(idx)
        picked.extend(idx[:per_class])
    out = docs.iloc[sorted(picked)].reset_index(drop=True)
    out["coverage_only"] = True
    return out


def build_cohort_report(docs) -> dict:
    """The cohort table: per doc_type x cohort counts + dup + messy summary.

    Reproducible from the pinned staged tree (``data/modernbert_training/
    stage``) — the report's inputs are the same parquet files the eval
    harness reads.
    """
    d = cohort_split(docs)
    d = window_cohort(d)
    table: dict[str, dict[str, int]] = {}
    for dt in sorted(d["doc_type"].astype(str).unique()):
        sub = d[d["doc_type"].astype(str) == dt]
        table[dt] = {
            "short": int((sub["cohort"] == "short").sum()),
            "long": int((sub["cohort"] == "long").sum()),
            "single_window": int((sub["window_cohort"] == "single-window").sum()),
            "multi_window": int((sub["window_cohort"] == "multi-window").sum()),
            "messy": int(sub["doc_text"].map(looks_messy).sum()),
        }
    return {
        "n_docs": int(len(d)),
        "by_doc_type": table,
        "duplicates": duplicate_groups(d),
        "note": ("cohort composition from the pinned staged tree; "
                 "single-window agreement is trivially 1.0 and must be "
                 "scored separately (#104)"),
    }