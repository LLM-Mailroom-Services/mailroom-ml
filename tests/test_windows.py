"""Windowing tests (port of corpus-eda test_prep.py window cases).

Tokenizer-dependent tests are skipped when ``transformers`` is absent
(train extra) — the module itself must remain importable without it.
No torch anywhere.
"""
from __future__ import annotations

import pytest

from conftest import requires_transformers
from mailroom_ml.config import MAX_TOKENS
from mailroom_ml.windows import estimate_tokens, window_document


@requires_transformers
@pytest.mark.train
def test_window_document_single_and_multi():
    short = window_document("t", "x" * 100)
    assert len(short) == 1 and short[0].startswith("t")
    long_text = "word " * 200_000  # ~800K chars >> 8,192 tokens
    wins = window_document("t", long_text)
    assert len(wins) > 1
    assert all(w.startswith("t\n\n") for w in wins)
    # overlap: consecutive windows share tail content
    assert wins[0][-50:] in wins[1]


@requires_transformers
@pytest.mark.train
def test_window_document_title_prefix_survives_verbatim():
    """The raw title string is re-attached verbatim — never clamped away."""
    title = "Quarterly Risk Call — Subject Line"
    wins = window_document(title, "word " * 200_000)
    assert len(wins) > 1
    assert all(w.startswith(title + "\n\n") for w in wins)
    # every window re-tokenizes within the model budget WITH default specials
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    assert all(len(tok(w)["input_ids"]) <= MAX_TOKENS for w in wins)


@requires_transformers
@pytest.mark.train
def test_window_document_without_title():
    wins = window_document("", "word " * 200_000)
    assert len(wins) > 1
    assert all(not w.startswith("\n\n") for w in wins)


@requires_transformers
@pytest.mark.train
def test_window_document_short_doc_single_window():
    # at or under budget: exactly one window, full text, title attached
    body = "This is a short document. " * 20
    wins = window_document("Short", body)
    assert len(wins) == 1
    assert wins[0].startswith("Short\n\n")
    assert body in wins[0]


@requires_transformers
@pytest.mark.train
def test_window_document_runtime_error_when_transformers_absent(monkeypatch):
    # guard path: force the cached tokenizer to None -> clear RuntimeError
    import mailroom_ml.windows as windows_mod

    monkeypatch.setattr(windows_mod, "_tokenizer", lambda: None)
    with pytest.raises(RuntimeError):
        window_document("t", "x" * 100)


def test_estimate_tokens_heuristic():
    assert estimate_tokens("x" * 100) == 25  # chars/4
    assert estimate_tokens("") == 1  # floor of 1
    assert estimate_tokens("a" * 7) == 2  # rounds, never 0
