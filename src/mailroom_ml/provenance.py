"""Provenance: sha256 helpers + the byte-deterministic build manifest.

The manifest mirrors the committed corpus-eda format (Mailroom-Corpus-EDA
commit cf096fa) — same line structure, no timestamps — and cites the
working corpus copy ``FINETUNE_REPO @ FINETUNE_REVISION``.  A rebuild on the
same pinned inputs reproduces the same bytes.
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter
from pathlib import Path

from mailroom_ml.config import (
    CANONICAL_REPO,
    CANONICAL_REVISION,
    FINETUNE_REPO,
    FINETUNE_REVISION,
    MAX_TOKENS,
    MODEL_ID,
    RANDOM_STATE,
    WINDOW_OVERLAP_TOKENS,
)

__all__ = [
    "sha256_bytes",
    "sha256_file",
    "clean_sha256",
    "atomic_write_text",
    "build_manifest",
    "record_manifest_sha",
]


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of raw bytes (canonical content hash for parquet files)."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes — the value byte-verified against the Hub."""
    return sha256_bytes(path.read_bytes())


def clean_sha256(value) -> str:
    """Normalize a ``content_sha256`` cell: null/blank/pseudo-hash -> ``""``.

    ``str(None)`` / ``str(float("nan"))`` would mint the pseudo-hashes
    ``"None"`` / ``"nan"`` and every hash-less row would then "match" every
    other one.  A missing hash is opaque (empty), never a value.
    """
    if value is None:
        return ""
    try:
        if bool(value != value):  # NaN / pd.NA-safe null check
            return ""
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return "" if s.lower() in {"nan", "none", "<na>"} else s


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` (UTF-8, no newline translation) via tmp + ``os.replace``.

    A crash/interrupt mid-write must never leave a truncated sidecar
    (``labels.json`` / ``manifest.txt`` ...) where a good one used to be: the
    bytes land in a sibling temp file first and replace the target atomically.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_bytes(text.encode("utf-8"))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def build_manifest(docs, counts: dict, maps: dict) -> str:
    """Byte-deterministic build manifest (no timestamps — determinism law).

    ``docs`` is the documents DataFrame, ``counts`` the stage stats dict
    (documents/windows per split), ``maps`` the per-head label maps; the
    publish commit (not the manifest) carries the time.
    """
    # lazy import: dataset imports this module at load time
    from mailroom_ml.dataset import filename_leak_audit

    types = Counter(docs["doc_type"])
    strata = Counter(zip(docs["doc_type"], docs["subclass"], strict=False))
    n_windows = sum(counts.get("windows", {}).values())
    # Computed from the data, never asserted: the split sizes and the label-leak
    # gate figures in the manifest must be the ones this very build produced.
    n_test = int((docs["split"] == "test").sum()) if "split" in docs.columns else 0
    leak = filename_leak_audit(docs)
    # NOTE: no built_utc timestamp — the manifest must be byte-identical
    # across rebuilds (determinism law); the publish commit carries the time.
    return f"""mailroom-modernbert-training manifest
==================================================
source_repo      : {FINETUNE_REPO} @ {FINETUNE_REVISION}
eval_corpus_pin  : {CANONICAL_REPO} @ {CANONICAL_REVISION} (immutable eval harness)
source_config    : ground_truth (labels) + default (doc_text), joined on filename
model            : {MODEL_ID} (max_tokens={MAX_TOKENS}, overlap={WINDOW_OVERLAP_TOKENS})
rows_total       : {len(docs)} ({dict(sorted(types.items()))})
rows_by_config   : documents {dict(counts.get("documents", {}))}; windows {dict(counts.get("windows", {}))} ({n_windows} total)
strata           : {len(strata)} (doc_type x canonical subclass)
split_rule       : corpus train -> 90/10 stratified train/validation (by doc_type,
                    seed {RANDOM_STATE}, RandomState shuffle); corpus test ({n_test})
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
                    the filename fallback was removed in the pre-flight clean
                    — it leaked the label: 42.8% of filenames carried the
                    subclass token, 65.9% of rows had title==filename)
leak_audit       : title==filename {leak["title_eq_filename"]}, title==filename-stem {leak["title_eq_filename_stem"]}, filename-shaped titles {leak["title_looks_like_filename"]} (gate in
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
