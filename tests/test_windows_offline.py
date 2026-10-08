"""Offline windowing tests: a deterministic fake tokenizer, no Hub, no torch.

The real ModernBERT tokenizer needs the network (see ``test_windows.py``);
the windowing ARITHMETIC (single-window budget, stride, coverage, overlap,
termination, tokenizer-cache hygiene) does not.  The fake tokenizer is
whitespace-word level, compositional, and honors ``add_special_tokens`` by
adding 2 specials ([CLS]/[SEP]) like the real one.
"""
from __future__ import annotations

import sys

import pytest

import mailroom_ml.windows as windows_mod
from mailroom_ml.windows import window_document

SPECIALS = (1, 2)  # [CLS], [SEP]
_FIRST_WORD_ID = 10


class FakeTokenizer:
    """Word-level tokenizer with a growing vocab; counts calls (runaway guard)."""

    def __init__(self, max_calls: int = 200_000):
        self._ids: dict[str, int] = {}
        self._words: dict[int, str] = {}
        self.calls = 0
        self.max_calls = max_calls

    def _bump(self):
        self.calls += 1
        if self.calls > self.max_calls:
            raise RuntimeError("runaway windowing loop (no termination)")

    def __call__(self, text, add_special_tokens=True):
        self._bump()
        ids = []
        for w in text.split():
            if w not in self._ids:
                self._ids[w] = _FIRST_WORD_ID + len(self._ids)
                self._words[self._ids[w]] = w
            ids.append(self._ids[w])
        if add_special_tokens:
            ids = [SPECIALS[0], *ids, SPECIALS[1]]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        self._bump()
        return " ".join(self._words[i] for i in ids if i >= _FIRST_WORD_ID)


@pytest.fixture
def fake_tok(monkeypatch):
    tok = FakeTokenizer()
    monkeypatch.setattr(windows_mod, "_tokenizer", lambda: tok)
    return tok


def _words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def _body_words(window: str, n_prefix: int) -> list[str]:
    return window.split()[n_prefix:]


# ---------------------------------------------------------------------------
# single-window budget counts the 2 specials (bug 6)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("n_tokens", [97, 98, 99, 100])
def test_single_window_shortcut_counts_specials(fake_tok, version, n_tokens):
    """A doc of max_tokens-1/-2 content tokens used to take the single-window
    shortcut and become a max_tokens+1 / +2 window (encode ValueError)."""
    max_tokens, overlap = 100, 10
    wins = window_document("", _words(n_tokens), max_tokens, overlap,
                           version=version, filename="")
    for w in wins:
        assert len(fake_tok(w)["input_ids"]) <= max_tokens
    if n_tokens > max_tokens - 2:
        assert len(wins) > 1


def test_single_window_exactly_at_budget_stays_single(fake_tok):
    wins = window_document("", _words(98), 100, 10)  # 98 + 2 specials == 100
    assert wins == [_words(98)]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_boundary_doc_every_window_fits_with_specials(fake_tok, version):
    max_tokens = 100
    for n in (max_tokens - 1, max_tokens, max_tokens + 1):
        wins = window_document("T0 T1", _words(n), max_tokens, 10,
                               version=version, filename="f.pdf")
        assert all(len(fake_tok(w)["input_ids"]) <= max_tokens for w in wins)


# ---------------------------------------------------------------------------
# no truncation: every id covered, configured overlap kept (bug 7)
# ---------------------------------------------------------------------------

def _assert_covered_with_overlap(wins, doc_words, n_prefix, overlap):
    bodies = [_body_words(w, n_prefix) for w in wins]
    seen = {w for b in bodies for w in b}
    # (v1 windows the decorated title+body ids, so the title's own tokens also
    # appear in the first window(s); coverage is about the DOCUMENT's tokens.)
    assert set(doc_words) <= seen, "tokens silently dropped between windows"
    for prev, nxt in zip(bodies, bodies[1:], strict=False):
        assert nxt[:overlap] == prev[-overlap:], "overlap shrank or went negative"
    assert bodies[-1][-1] == doc_words[-1]


@pytest.mark.parametrize("title_words", [0, 5, 40, 60])
def test_v1_windows_cover_whole_document_with_overlap(fake_tok, title_words):
    title = _words(title_words, "T")
    doc = _words(1000)
    overlap = 10
    wins = window_document(title, doc, 100, overlap)
    assert len(wins) > 1
    assert all(len(fake_tok(w)["input_ids"]) <= 100 for w in wins)
    _assert_covered_with_overlap(wins, doc.split(), title_words, overlap)


@pytest.mark.parametrize("title_words", [0, 5, 20])
def test_v2_windows_cover_whole_document_with_overlap(fake_tok, title_words):
    title = _words(title_words, "T")
    doc = _words(1000)
    overlap = 10
    wins = window_document(title, doc, 100, overlap, version="v2",
                           filename="f.pdf")
    assert len(wins) > 1
    assert all(len(fake_tok(w)["input_ids"]) <= 100 for w in wins)
    # v2 prefix: [FILE_NAME] f.pdf [TITLE] <title> [WINDOW_INDEX] n
    prefix = windows_mod.decorate_window(title, "", version="v2",
                                         filename="f.pdf", window_index=0)
    n_prefix = len(prefix.split())
    _assert_covered_with_overlap(wins, doc.split(), n_prefix, overlap)


def test_stride_never_negative_with_long_title(fake_tok):
    """body_budget (38) < old step (max_tokens - overlap = 90): the old slide
    skipped 52 tokens between windows."""
    title = _words(60, "T")
    doc = _words(400)
    wins = window_document(title, doc, 100, 10)
    bodies = [_body_words(w, 60) for w in wins]
    assert set(doc.split()) <= {w for b in bodies for w in b}


# ---------------------------------------------------------------------------
# termination on an enormous title (bug 5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title_words", [97, 98, 150])
def test_v1_terminates_when_title_fills_context(fake_tok, title_words):
    """The trim loop had no `len(body_ids) <= 8` guard in v1: a title that
    alone exceeds the budget spun forever (FakeTokenizer raises on runaway)."""
    wins = window_document(_words(title_words, "T"), _words(30), 100, 10)
    assert wins  # over-context windows are returned; encode_inputs rejects


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_title_leaving_no_room_past_overlap_routes_llm(fake_tok, version):
    """body_budget <= overlap: the slide could only advance ~1 token per
    window (a 2,000-token doc -> ~2,000 windows).  One over-context window
    is returned instead, which encode_inputs rejects -> LLM route."""
    wins = window_document(_words(85, "T"), _words(2000), 100, 16,
                           version=version, filename="f.pdf")
    assert len(wins) == 1


def test_v2_terminates_when_prefix_fills_context(fake_tok):
    wins = window_document(_words(150, "T"), _words(30), 100, 10,
                           version="v2", filename="f.pdf")
    assert wins


# ---------------------------------------------------------------------------
# tokenizer cache hygiene (bug 4)
# ---------------------------------------------------------------------------

def test_tokenizer_import_error_is_cached_absent(monkeypatch):
    monkeypatch.setattr(windows_mod, "_TOKENIZER_CACHE", None)
    monkeypatch.setitem(sys.modules, "transformers", None)  # import -> ImportError
    assert windows_mod._tokenizer() is None
    assert windows_mod._TOKENIZER_CACHE is False


def test_transient_hub_error_is_not_cached(monkeypatch):
    transformers = pytest.importorskip("transformers")
    calls = []

    class _Tok:
        pass

    def _from_pretrained(_model_id):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("Hub unreachable")
        return _Tok()

    monkeypatch.setattr(windows_mod, "_TOKENIZER_CACHE", None)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        staticmethod(_from_pretrained))
    with pytest.raises(OSError):
        windows_mod._tokenizer()
    assert windows_mod._TOKENIZER_CACHE is None  # not poisoned with False
    tok = windows_mod._tokenizer()  # retried and recovered
    assert isinstance(tok, _Tok)
    assert windows_mod._tokenizer() is tok  # success IS cached
    assert len(calls) == 2
