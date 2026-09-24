"""Token-level windowing for ModernBERT (port of ``modernbert/prep.py``).

Every published window is ``title + "\\n\\n" + window_text`` re-verified to
re-tokenize within the model's context WITH the tokenizer's default specials
— the BPE decode->re-encode clamp.  Deterministic for a pinned tokenizer
(model snapshot ``answerdotai/ModernBERT-base``).

``transformers`` is an optional train extra: it is imported lazily inside
``window_document`` and a clear ``RuntimeError`` is raised when absent
(``uv sync`` WITHOUT ``--extra train`` still imports/tests this module).
"""
from __future__ import annotations

from collections.abc import Callable

from mailroom_ml.config import (
    CHARS_PER_TOKEN,
    MAX_TOKENS,
    MODEL_ID,
    WINDOW_OVERLAP_TOKENS,
)

__all__ = ["window_document", "estimate_tokens"]

_TOKENIZER_CACHE: Callable | None = None


def _tokenizer():
    """ModernBERT tokenizer (cached); None when transformers is absent.

    Mirrors the committed corpus-eda helper: a failed lazy import caches
    ``None`` so the guard costs nothing on repeated calls, and
    ``model_max_length`` is raised so per-call context warnings are silenced
    (we chunk below 8,192 tokens ourselves).
    """
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(MODEL_ID)
            # full-doc tokenization for windowing is intentionally longer than
            # the model context — silence the per-call warning (we chunk).
            tok.model_max_length = 1 << 30
            # Byte-compat with the published training set (transformers 5.x
            # drift): transformers >= 5 added a guard that SKIPS the legacy
            # clean_up_tokenization post-processing for BPE tokenizers, so
            # decode() would emit "below ." where the committed build (and
            # the published windows) emit "below.".  Re-enable the legacy
            # cleanup so rebuilds stay byte-identical; setting this attribute
            # is a no-op attribute on older transformers (attribute set).
            tok.clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output = True  # noqa: E501
            _TOKENIZER_CACHE = tok
        except Exception:  # noqa: BLE001 — train deps not installed
            _TOKENIZER_CACHE = False
    return _TOKENIZER_CACHE or None


def window_document(title: str, doc_text: str, max_tokens: int = MAX_TOKENS,
                    overlap: int = WINDOW_OVERLAP_TOKENS) -> list[str]:
    """Token-level windows: ``title + "\\n\\n" + window`` per chunk.

    The full input (title + body) is tokenized once; chunks of
    ``max_tokens`` with ``overlap``-token overlap are decoded back to text.
    BPE decode->re-encode is not idempotent, so each window's text is
    re-tokenized and clamped to ``max_tokens`` — every published window is
    guaranteed to fit the model's context.  Deterministic for a pinned
    tokenizer.  A doc at or under budget yields a single window (title +
    head).

    Raises ``RuntimeError`` when ``transformers`` is not installed (train
    extras are optional for this package — stage-only builds that skip
    windows, or imports, never need them).
    """
    tok = _tokenizer()
    if tok is None:
        raise RuntimeError(
            "transformers not installed — windowing needs the ModernBERT "
            f"tokenizer ({MODEL_ID}); install the train extras, e.g. "
            "'uv sync --extra train'"
        )
    full = f"{title}\n\n{doc_text}" if title else doc_text
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if not ids:
        return [full]
    if len(ids) <= max_tokens:
        return [full]
    # headroom for the title + the tokenizer's default specials (<s></s>):
    # every published window re-tokenizes to <= max_tokens WITH specials.
    n_title = len(tok(title, add_special_tokens=False)["input_ids"]) if title else 0
    body_budget = max_tokens - 2 - n_title
    step = max_tokens - overlap
    windows = []
    for start in range(0, len(ids), step):
        chunk = ids[start:start + max_tokens]
        body = tok.decode(chunk, skip_special_tokens=True)
        # BPE decode->re-encode is not idempotent: clamp the BODY (never the
        # title — the raw title string is re-attached verbatim) so the
        # decorated window fits the model context with specials.
        re_ids = tok(body, add_special_tokens=False)["input_ids"]
        if len(re_ids) > body_budget:
            body = tok.decode(re_ids[:body_budget], skip_special_tokens=True)
        decorated = f"{title}\n\n{body}" if title else body
        # BPE is not compositional across the "\n\n" boundary (the separator
        # can add 1-2 tokens): verify the DECORATED string and trim the body
        # tail until it fits — deterministic, terminates (body shrinks).
        while len(tok(decorated)["input_ids"]) > max_tokens - 2:
            body_ids = tok(body, add_special_tokens=False)["input_ids"]
            body = tok.decode(body_ids[:-8], skip_special_tokens=True)
            decorated = f"{title}\n\n{body}" if title else body
        windows.append(decorated)
        if start + max_tokens >= len(ids):
            break
    return windows


def estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Chars/4 heuristic token estimate (tiktoken-o200k-accurate at scale)."""
    return max(1, int(round(len(text) / chars_per_token)))
