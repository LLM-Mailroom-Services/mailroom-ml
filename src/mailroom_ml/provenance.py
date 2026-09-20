"""Provenance: sha256 helpers + the byte-deterministic build manifest.

The manifest mirrors the committed corpus-eda format (Mailroom-Corpus-EDA
commit cf096fa) — same line structure, no timestamps — and cites the
working corpus copy ``FINETUNE_REPO @ FINETUNE_REVISION``.  A rebuild on the
same pinned inputs reproduces the same bytes.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from mailroom_ml.config import (
    FINETUNE_REPO,
    FINETUNE_REVISION,
    MAX_TOKENS,
    MODEL_ID,
    RANDOM_STATE,
    WINDOW_OVERLAP_TOKENS,
)

__all__ = ["sha256_bytes", "sha256_file", "build_manifest", "record_manifest_sha"]


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of raw bytes (canonical content hash for parquet files)."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes — the value byte-verified against the Hub."""
    return sha256_bytes(path.read_bytes())


def build_manifest(docs, counts: dict, maps: dict) -> str:
    """Byte-deterministic build manifest (no timestamps — determinism law).

    ``docs`` is the documents DataFrame, ``counts`` the stage stats dict
    (documents/windows per split), ``maps`` the per-head label maps; the
    publish commit (not the manifest) carries the time.
    """
    types = Counter(docs["doc_type"])
    strata = Counter(zip(docs["doc_type"], docs["subclass"], strict=False))
    n_windows = sum(counts.get("windows", {}).values())
    # NOTE: no built_utc timestamp — the manifest must be byte-identical
    # across rebuilds (determinism law); the publish commit carries the time.
    return f"""mailroom-modernbert-training manifest
==================================================
source_repo      : {FINETUNE_REPO} @ {FINETUNE_REVISION}
source_config    : ground_truth (labels) + default (doc_text), joined on filename
model            : {MODEL_ID} (max_tokens={MAX_TOKENS}, overlap={WINDOW_OVERLAP_TOKENS})
rows_total       : {len(docs)} ({dict(sorted(types.items()))})
rows_by_config   : documents {dict(counts.get("documents", {}))}; windows {dict(counts.get("windows", {}))} ({n_windows} total)
strata           : {len(strata)} (doc_type x canonical subclass)
split_rule       : corpus train -> 90/10 stratified train/validation (by doc_type,
                    seed {RANDOM_STATE}, RandomState shuffle); corpus test (323)
                    held out entirely — never touches training
subclass_norm    : llm-dojo-scoring normalize_corpus_subclass (DMR-066),
                    vendored in mailroom_ml/labels.py (Service/service,
                    Co_Branding/co_branding, Joint Venture _ Filing -> joint_venture)
text_clerk       : llm-dojo-scoring deterministic_normalize (DMR-066),
                    vendored in mailroom_ml/normalize.py — the SAME clerk
                    llm-mailroom apply_intake runs before the classifier, so
                    training input is byte-representative of inference input
                    (NFC, newline unify, NBSP, zero-width, C0 controls, hyphen
                    unwrap, blank-run collapse, horizontal collapse, trim)
title_rule       : subject -> exhibit_description -> EMPTY (semantic-only;
                    the filename fallback was removed 2026-09-20 — it leaked
                    the label: 42.8% of filenames carried the subclass token,
                    65.9% of rows had title==filename)
leak_audit       : title==filename 0, filename-shaped titles 0 (gate in
                    mailroom_ml.dataset.filename_leak_audit + verify_stage)
heads            : doc_type (5 + unknown) + per-class subclass heads
                    ({", ".join(f"{k}: {len(v['labels'])}" for k, v in maps.items())})
builder          : mailroom_ml.dataset.stage @ mailroom-ml
"""


def record_manifest_sha(manifest: str) -> str:
    """Hex sha256 of the manifest text — the recorded build fingerprint.

    ``stage()`` stores this value in its stats dict (``manifest_sha256``)
    and the publish CLI byte-verifies it against the Hub sidecar.
    """
    return sha256_bytes(manifest.encode("utf-8"))
