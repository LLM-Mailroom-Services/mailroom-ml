"""Token-level windowing for ModernBERT (port of ``modernbert/prep.py``).

Every published v1 window is ``title + "\\n\\n" + window_text`` re-verified
to re-tokenize within the model's context WITH the tokenizer's default
specials — the BPE decode->re-encode clamp.  Deterministic for a pinned
tokenizer (model snapshot ``answerdotai/ModernBERT-base``).

v2 (#29) prepends tagged ``[FILE_NAME]`` / ``[TITLE]`` / ``[WINDOW_INDEX]``
metadata.  It is NEVER mixed into the published v1 Hub revision; switching
is an explicit trainer flag persisted in ``summary.json``.

``transformers`` is an optional train extra: it is imported lazily inside
``window_document`` and a clear ``RuntimeError`` is raised when absent
(``uv sync`` WITHOUT ``--extra train`` still imports/tests this module).
"""
from __future__ import annotations

from collections.abc import Callable

from mailroom_ml.config import (
    CHARS_PER_TOKEN,
    INPUT_CONSTRUCTION_VERSION,
    INPUT_CONSTRUCTION_VERSIONS,
    MAX_TOKENS,
    MODEL_ID,
    WINDOW_OVERLAP_TOKENS,
)

__all__ = ["decorate_window", "window_document", "estimate_tokens"]

_TOKENIZER_CACHE: Callable | None = None


def _tokenizer():
    """ModernBERT tokenizer (cached); None when transformers is absent.

    Mirrors the committed corpus-eda helper: a failed lazy import caches
    ``False`` so the guard costs nothing on repeated calls, and
    ``model_max_length`` is raised so per-call context warnings are silenced
    (we chunk below 8,192 tokens ourselves).

    Only a missing ``transformers`` (``ImportError``) is cached as absent.
    Any other failure (Hub/network error, bad snapshot) is transient: it
    propagates uncached so the next call retries instead of poisoning the
    process with "no tokenizer" forever; ``classify_document`` wraps the
    raise into a fail-open ``bert_error`` -> LLM route.
    """
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        try:
            from transformers import AutoTokenizer
        except ImportError:  # train extras not installed
            _TOKENIZER_CACHE = False
            return None
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
    return _TOKENIZER_CACHE or None


def decorate_window(
    title: str,
    body: str,
    *,
    version: str = INPUT_CONSTRUCTION_VERSION,
    filename: str = "",
    window_index: int = 0,
) -> str:
    """Apply the input-construction prefix (#29).

    v1 (default, Hub-published): ``title + "\\n\\n" + body`` when title is
    non-empty, else the body alone — byte-compatible with
    ``TRAINING_DATA_REVISION``.  v2 prepends tagged metadata lines
    ``[FILE_NAME]`` / ``[TITLE]`` / ``[WINDOW_INDEX]``.  Unknown versions
    raise — never silently fall through to a mixed format.
    """
    if version not in INPUT_CONSTRUCTION_VERSIONS:
        raise ValueError(
            f"unknown input construction version {version!r}; "
            f"known: {INPUT_CONSTRUCTION_VERSIONS}"
        )
    if version == "v1":
        return f"{title}\n\n{body}" if title else body
    return (
        f"[FILE_NAME] {filename}\n"
        f"[TITLE] {title}\n"
        f"[WINDOW_INDEX] {window_index}\n\n"
        f"{body}"
    )


def window_document(title: str, doc_text: str, max_tokens: int = MAX_TOKENS,
                    overlap: int = WINDOW_OVERLAP_TOKENS,
                    *,
                    version: str = INPUT_CONSTRUCTION_VERSION,
                    filename: str = "") -> list[str]:
    """Token-level windows: decorated ``title + body`` per chunk.

    v1 follows the published algorithm's construction (tokenizes the
    decorated full string, then re-attaches the raw title) but slides over
    the whole document with no truncation (see ``_window_document_v1``).  v2
    tokenizes the body and re-applies the tagged prefix per window.  Unknown
    versions raise.

    Raises ``RuntimeError`` when ``transformers`` is not installed (train
    extras are optional for this package — stage-only builds that skip
    windows, or imports, never need them).  A Hub/network
    failure while loading the tokenizer propagates (uncached; retried on the
    next call) instead of masquerading as "transformers not installed".
    """
    if version not in INPUT_CONSTRUCTION_VERSIONS:
        raise ValueError(
            f"unknown input construction version {version!r}; "
            f"known: {INPUT_CONSTRUCTION_VERSIONS}"
        )
    tok = _tokenizer()
    if tok is None:
        raise RuntimeError(
            "transformers not installed — windowing needs the ModernBERT "
            f"tokenizer ({MODEL_ID}); install the train extras, e.g. "
            "'uv sync --extra train'"
        )
    if version == "v1":
        return _window_document_v1(tok, title, doc_text, max_tokens, overlap)
    return _window_document_v2(
        tok, title, doc_text, max_tokens, overlap, filename=filename)


def _slide_windows(tok, ids: list[int], *, body_budget: int, max_tokens: int,
                   overlap: int, decorate: Callable[[str, int], str]
                   ) -> list[str]:
    """Slide over ``ids`` so every token lands in some window (no truncation).

    Each chunk is ``body_budget`` ids wide (the room left AFTER the title /
    tagged prefix and the 2 specials), then clamped and trimmed toward the
    ``max_tokens`` budget including specials. A window can remain over budget
    when at most eight body tokens remain; the caller must validate its
    length. The decode->re-encode clamp and the tail trim can both shorten a chunk, so the next window
    starts from the REAL covered end minus ``overlap`` (never from a nominal
    ``start + max_tokens``): the cut tail is re-covered, not dropped, and the
    overlap can never go negative.  Always advances >= 1 id, so it
    terminates.
    """
    windows: list[str] = []
    start = 0
    while True:
        chunk = ids[start:start + body_budget]
        body = tok.decode(chunk, skip_special_tokens=True)
        cut = False
        # BPE decode->re-encode is not idempotent: clamp the BODY (never the
        # title — the raw title string is re-attached verbatim) so the
        # decorated window fits the model context with specials.
        re_ids = tok(body, add_special_tokens=False)["input_ids"]
        if len(re_ids) > body_budget:
            body = tok.decode(re_ids[:body_budget], skip_special_tokens=True)
            cut = True
        decorated = decorate(body, len(windows))
        # BPE is not compositional across the "\n\n" boundary (the separator
        # can add 1-2 tokens): verify the DECORATED string and trim the body
        # tail until it fits — deterministic, terminates (body shrinks; the
        # guard stops a title/prefix that alone fills the budget).
        while len(tok(decorated)["input_ids"]) > max_tokens - 2:
            body_ids = tok(body, add_special_tokens=False)["input_ids"]
            if len(body_ids) <= 8:
                break
            body = tok.decode(body_ids[:-8], skip_special_tokens=True)
            cut = True
            decorated = decorate(body, len(windows))
        windows.append(decorated)
        covered_end = start + len(chunk)
        if cut:
            kept = len(tok(body, add_special_tokens=False)["input_ids"])
            covered_end = start + max(1, min(len(chunk), kept))
        if covered_end >= len(ids):
            break
        start = max(covered_end - overlap, start + 1)
    return windows


def _window_document_v1(tok, title: str, doc_text: str,
                        max_tokens: int, overlap: int) -> list[str]:
    """v1 construction (``title`` + ``\\n\\n`` + body), no longer byte-identical to
    the windows of the published TRAINING_DATA_REVISION windower.

    Changed for the no-truncation doctrine (#85): the published windower
    (a) took the single-window shortcut at ``max_tokens`` content tokens even
    though the tokenizer adds 2 specials (an 8,191/8,192-token doc became an
    over-context window that ``encode_inputs`` rejected), and (b) slid by
    ``max_tokens - overlap`` over chunks that were then clamped to the (much
    smaller, title-dependent) body budget and trimmed, silently dropping the
    tail tokens between windows and, with long titles, a negative overlap.
    Windows now slide by the real body budget from the real covered end, so
    the whole document is covered with the configured overlap.  Single-window
    docs are unchanged (bar the 2-token boundary above); the first window of
    a long doc is the same, later windows start at different offsets.  A
    training-set rebuild with this windower needs a new pinned revision.

    If the title leaves a body budget no greater than ``max(overlap, 0)``,
    return the full text in one potentially over-context window for the
    caller to reject. Tokenizer errors propagate.
    """
    full = f"{title}\n\n{doc_text}" if title else doc_text
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if not ids:
        return [full]
    # same with-specials budget as the multi-window path
    if len(ids) <= max_tokens - 2:
        return [full]
    # headroom for the title + the tokenizer's default specials (<s></s>):
    # every window re-tokenizes to <= max_tokens WITH specials.
    n_title = len(tok(title, add_special_tokens=False)["input_ids"]) if title else 0
    body_budget = max_tokens - 2 - n_title
    if body_budget <= max(overlap, 0):
        # the title (nearly) fills the context: a window carries no body
        # beyond the overlap, so the slide would crawl ~1 token per window
        # (thousands of windows).  Hand back the over-context text —
        # encode_inputs raises and the caller routes LLM (never truncate).
        return [full]
    return _slide_windows(
        tok, ids, body_budget=body_budget, max_tokens=max_tokens,
        overlap=overlap,
        decorate=lambda body, _i: f"{title}\n\n{body}" if title else body)


def _window_document_v2(tok, title: str, doc_text: str,
                        max_tokens: int, overlap: int, *,
                        filename: str) -> list[str]:
    """Tagged-prefix windower.  Not mixed into the v1 Hub revision.

    Same coverage fix as v1: the single-window shortcut uses the
    with-specials budget and the slide follows the real body budget / covered
    end (see :func:`_window_document_v1`).

    If the tagged prefix leaves a body budget no greater than
    ``max(overlap, 0)``, return one full decorated window, even if it exceeds
    the context limit. Tokenizer errors propagate.
    """
    ids = tok(doc_text, add_special_tokens=False)["input_ids"]
    full = decorate_window(title, doc_text, version="v2",
                           filename=filename, window_index=0)
    if not ids:
        return [full]
    prefix0 = decorate_window(title, "", version="v2", filename=filename,
                              window_index=0)
    n_prefix = len(tok(prefix0, add_special_tokens=False)["input_ids"])
    if n_prefix + len(ids) <= max_tokens - 2:
        return [full]
    body_budget = max_tokens - 2 - n_prefix
    if body_budget <= max(overlap, 0):
        return [full]  # prefix (nearly) fills the context -> LLM (see v1)
    return _slide_windows(
        tok, ids, body_budget=body_budget, max_tokens=max_tokens,
        overlap=overlap,
        decorate=lambda body, i: decorate_window(
            title, body, version="v2", filename=filename, window_index=i))


def estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Chars/4 heuristic token estimate (tiktoken-o200k-accurate at scale)."""
    return max(1, int(round(len(text) / chars_per_token)))
