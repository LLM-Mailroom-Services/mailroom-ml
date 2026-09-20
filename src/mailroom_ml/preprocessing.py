"""Preprocessing: title construction + the deterministic-normalizer seam.

Port of ``modernbert/prep.py`` title logic (Mailroom-Corpus-EDA commit
cf096fa).  The full ``doc_text`` is consumed **as published by the corpus** —
the pipeline's ``apply_intake`` step (intake sorter) runs its deterministic
normalizer BEFORE the classifier, so this package never re-normalizes stored
text; ``normalize_text`` documents that seam (see its docstring).
"""
from __future__ import annotations

from typing import Any

__all__ = ["build_title", "normalize_text"]


def build_title(row: dict[str, Any]) -> str:
    """Title-wins signal: subject -> exhibit_description -> **empty**.

    Mirrors the corpus's own title conventions (the sorter prompt's
    title-wins doctrine): correspondence carries real subject lines, EDGAR
    exhibits carry exhibit descriptions.

    LEAK FIX (2026-09-20 pre-flight audit): the previous rule fell back to
    ``filename``.  That leaked the label — 42.8% of documents carry the
    subclass token in their filename (insurance_claim 86%, contract 76%,
    merger_agreement 100% of doc_type), and 65.9% of documents had
    ``title == filename``.  Because the window input is
    ``title + "\\n\\n" + body``, the model could read the answer off the
    filename (insurance_claim subclass macro-F1 was a suspicious 1.0).  A
    real pipeline filename does not encode the class, so the fallback is a
    train/inference mismatch as well as a leak.  No semantic title -> empty
    string -> the window is body-only.
    """
    for key in ("subject", "exhibit_description"):
        v = str((row.get("metadata") or {}).get(key) or "").strip()
        if v:
            return v
    return ""


def normalize_text(text: str, **kwargs: Any) -> str:
    """STUB — documents the deterministic-normalizer seam (do not call).

    The intake pipeline's ``apply_intake`` runs its deterministic text
    normalizer BEFORE the classifier and stores the normalized artifact;
    training/eval consume that stored text verbatim.  This module therefore
    only documents/imports the contract:

    - normalization is deterministic (same input -> same output, pinned
      revision), happens upstream in the pipeline, and is **never** applied
      here at training time;
    - re-normalizing at train/eval time would silently drift from the
      inference path's inputs (normalization mismatch = eval illusion).

    Raises ``NotImplementedError`` so any accidental call is loud instead of
    silently returning un-normalized text.
    """
    raise NotImplementedError(
        "normalize_text is a documented seam, not an implementation: the "
        "pipeline's apply_intake normalizes before the classifier; training "
        "consumes stored normalized text verbatim (see module docstring)."
    )
