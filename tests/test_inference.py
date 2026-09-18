"""Inference-layer tests (mailroom-issues #85 M3, plan §4 composite score).

Hermetic: stub bundles with injected logits — no network, no model
downloads, no GPU.  Covers the plurality merge, the composite route score
S = p*a*m (plan D4), the unknown/abstention path (#51/#43), the fast-path
gate truth table (context-fit, vocabulary, support, confidence/agreement/
margin) and fail-open behavior (D10).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mailroom_ml.config import (
    ABSTAIN_UNKNOWN_CLASS,
    BERT_INTAKE_MAX_CHARS,
)
from mailroom_ml.inference import (
    BundleLoadError,
    BundleUnavailable,
    ModelBundle,
    classify_document,
    classify_windows,
    encode_inputs,
    load_bundle,
    resolve_model_dir,
    window_titles,
)

# ---------------------------------------------------------------------------
# Stub bundle factory (tokenizer + logits injected — no deps beyond numpy)
# ---------------------------------------------------------------------------

DOC_TYPES = ("contract", "merger_agreement", "corporate_record",
             "correspondence", "insurance_claim", ABSTAIN_UNKNOWN_CLASS)
HEADS = {
    "doc_type": {"labels": list(DOC_TYPES),
                 "label2id": {k: i for i, k in enumerate(DOC_TYPES)},
                 "id2label": {str(i): k for i, k in enumerate(DOC_TYPES)}},
    "correspondence": {"labels": ["email", "letter", "notice",
                                  "attorney_demand"],
                       "label2id": {"email": 0, "letter": 1, "notice": 2,
                                    "attorney_demand": 3},
                       "id2label": {"0": "email", "1": "letter",
                                    "2": "notice", "3": "attorney_demand"}},
    "contract": {"labels": ["service", "license"],
                 "label2id": {"service": 0, "license": 1},
                 "id2label": {"0": "service", "1": "license"}},
}


class _FakeEnc:
    def __init__(self, ids):
        self.ids = ids


class _FakeTok:
    PAD = 0

    def __init__(self, word_ids: dict[str, int]):
        self._w = word_ids

    def encode_batch(self, texts):
        return [_FakeEnc([self._w.get(t, 1) for t in text.split()])
                for text in texts]

    def token_to_id(self, tok):
        return self._w.get(tok)


def _stub_bundle(predict_fn, *, maps=None, temperatures=None,
                 support=None, tokenizer=None, pad_id=0, **kw) -> ModelBundle:
    return ModelBundle(
        model_dir=Path("/stub"),
        maps=maps or HEADS,
        temperatures=temperatures or {},
        support_counts=support or {"correspondence": {"notice": 12}},
        tokenizer=tokenizer or _FakeTok({"<s>": 1, "</s>": 2}),
        pad_id=pad_id,
        model_kind="stub",
        predict_fn=predict_fn,
        **kw,
    )


def _logits_like(fn: str, head_vecs: dict[str, list[float]],
                 n_windows: int) -> dict[str, np.ndarray]:
    """One-hot-ish logits per head: vec per window replicated."""
    return {h: np.tile(np.array(v, dtype=np.float32), (n_windows, 1))
            for h, v in head_vecs.items()}


# ---------------------------------------------------------------------------
# Encoding / contract
# ---------------------------------------------------------------------------

def test_window_titles_v1_format():
    assert window_titles("T", ["a", "b"]) == ["T\n\na", "T\n\nb"]
    assert window_titles("", ["a"]) == ["a"]


def test_encode_inputs_pads_to_max():
    b = _stub_bundle(None)
    ids, mask = encode_inputs(b, ["hello world", "hi"], max_length=64)
    assert ids.shape == (2, 2)
    assert mask.tolist() == [[1, 1], [1, 0]]


def test_encode_inputs_never_truncates():
    b = _stub_bundle(None)
    with pytest.raises(ValueError):
        encode_inputs(b, ["x y z w v"], max_length=3)


# ---------------------------------------------------------------------------
# classify_windows — plurality merge + composite score
# ---------------------------------------------------------------------------

def test_plurality_merge_and_composite_score():
    """All windows vote correspondence/notice with high margins.

    S = p*a*m: p=0.9 (mean calib prob of the winner, temperature 1 keeps
    softmax of the injected logits), a=1.0, m=(p1 - p2) ~ 0.8 -> S ~ 0.72.
    """
    dt = [0.0, 0.0, 0.0, 10.0, 0.0, -5.0]  # correspondence wins massively
    corr = [0.0, 0.0, 9.0, 0.0]             # notice wins
    b = _stub_bundle(
        lambda ids, mask: _logits_like("x", {"doc_type": dt,
                                             "correspondence": corr}, 2))
    res = classify_windows(b, ["one window", "second window"])
    assert res["doc_type"] == "correspondence"
    assert res["subclass"] == "notice"
    assert res["agreement"] == 1.0
    assert res["margin"] > 0.7
    assert res["score"] == pytest.approx(res["calibrated_confidence"]
                                         * res["agreement"] * res["margin"],
                                         rel=1e-3)
    assert res["n_class_windows"] == 2


def test_unknown_abstention_never_scores_subclass():
    """All windows vote the inference-only unknown -> route llm, no subclass."""
    dt = [-5.0, 0.0, 0.0, 0.0, 0.0, 10.0]  # unknown index 5
    b = _stub_bundle(lambda ids, mask: _logits_like(
        "x", {"doc_type": dt, "correspondence": [0.0, 0.0, 9.0, 0.0]}, 1))
    res = classify_windows(b, ["abstain me"])
    assert res["doc_type"] == ABSTAIN_UNKNOWN_CLASS
    assert res["subclass"] is None
    assert res["route"] == "llm"
    assert "guard_failures" in res


# ---------------------------------------------------------------------------
# classify_document — the fast-path gate + fail-open
# ---------------------------------------------------------------------------

def _confident_doc_bundle():
    dt = [0.0, 0.0, 0.0, 10.0, 0.0, -6.0]
    corr = [0.0, 0.0, 10.0, 0.0]
    b = _stub_bundle(lambda ids, mask: _logits_like(
        "x", {"doc_type": dt, "correspondence": corr}, 1),
        support={"correspondence": {"notice": 12}})
    return b


def test_fast_path_high_confidence_doc():
    b = _confident_doc_bundle()
    res = classify_document(b, "Notice", "A notice demanding payment.",
                            window_texts=["notice text"])
    assert res["status"] == "ok"
    assert res["doc_type"] == "correspondence"
    assert res["subclass"] == "notice"
    assert res["route"] == "fast_path"
    assert res["reason"] == "fast_path"
    assert res["quality"]["coverage"] == 1.0
    assert res["quality"]["context_fit"] is True
    assert res["quality"]["triage_vocab_ok"] is True
    assert res["quality"]["sections_ok"] is True


def test_oversize_doc_routes_llm_never_truncates():
    b = _stub_bundle(lambda ids, mask: {})
    res = classify_document(b, "T", "x" * (BERT_INTAKE_MAX_CHARS + 1))
    assert res["route"] == "llm"
    assert res["reason"] == "oversize_chars"
    assert "oversize_chars" in res["guard_failures"]


def test_insufficient_authentic_support_fails_open():
    b = _stub_bundle(
        lambda ids, mask: _logits_like(
            "x", {"doc_type": [0.0, 0, 0, 10.0, 0, -6.0],
                  "correspondence": [0.0, 0.0, 10.0, 0.0]}, 1),
        support={"correspondence": {"notice": 2}})  # < ROUTE_MIN_AUTHENTIC_SUPPORT
    res = classify_document(b, "T", "body", window_texts=["w"])
    assert res["status"] == "ok"
    assert res["route"] == "llm"
    assert res["reason"] == "insufficient_support"


def test_low_confidence_routes_llm():
    """Winner prob must clear ROUTE_DOC_CONFIDENCE for the fast path."""
    dt = [2.0, 1.0, 1.0, 1.0, 1.0, 0.0]  # contract, but p ~ 0.23
    corr = [0.0, 0.0, 9.0, 0.0]
    b = _stub_bundle(lambda ids, mask: _logits_like(
        "x", {"doc_type": dt, "correspondence": corr}, 1))
    res = classify_document(b, "T", "body", window_texts=["w"])
    assert res["doc_type"] == "contract"
    assert res["route"] == "llm"
    assert res["reason"] == "gate_fail"
    assert any("doc_confidence" in f for f in res["guard_failures"])


def test_agreement_and_margin_gates():
    """Split votes across classes fail agreement; tight margins fail margin."""
    dt_a = [0.0, 0.0, 0.0, 10.0, 0.0, -5.0]
    dt_b = [0.0, 0.0, 0.0, 0.0, 10.0, -5.0]

    def _split(ids, mask):
        # first half of the batch votes correspondence, second half insurance
        n = ids.shape[0]
        rows_a = _logits_like("a", {"doc_type": dt_a,
                                    "correspondence": [0.0, 0.0, 9.0, 0.0]},
                              max(1, n // 2))
        rows_b = _logits_like("b", {"doc_type": dt_b,
                                    "correspondence": [0.0, 0.0, 9.0, 0.0]},
                              max(1, n - n // 2))
        return {h: np.vstack([rows_a[h], rows_b[h]]) for h in rows_a}

    b = _stub_bundle(_split)
    res = classify_document(b, "T", "body",
                            window_texts=["w1", "w2", "w3", "w4"])
    assert res["route"] == "llm"
    assert res["reason"] == "gate_fail"
    assert any("agreement" in f for f in res["guard_failures"])


def test_fail_open_on_broken_predict():
    """An exception inside the ML path degrades to route=llm (never raises)."""

    def _boom(ids, mask):
        raise RuntimeError("onnx session dead")

    b = _stub_bundle(_boom)
    res = classify_document(b, "T", "body", window_texts=["w"])
    assert res["status"] == "failure"
    assert res["route"] == "llm"
    assert res["reason"] == "bert_error"
    assert res["failure"]["type"] == "RuntimeError"
    assert "bert_error" in res["guard_failures"]


def test_abstaining_all_windows_routes_llm():
    dt = [-5.0, 0.0, 0.0, 0.0, 0.0, 10.0]
    b = _stub_bundle(lambda ids, mask: _logits_like(
        "x", {"doc_type": dt, "correspondence": [0.0, 0.0, 9.0, 0.0]},
        ids.shape[0]))
    res = classify_document(b, "T", "body", window_texts=["w", "w2"])
    assert res["doc_type"] == ABSTAIN_UNKNOWN_CLASS
    assert res["route"] == "llm"


# ---------------------------------------------------------------------------
# Model loading — resolution + clean failures
# ---------------------------------------------------------------------------

def test_resolve_model_dir_env_override(tmp_path, monkeypatch):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "labels.json").write_text("{}")
    good = tmp_path / "good"
    good.mkdir()
    (good / "labels.json").write_text("{}")
    monkeypatch.setenv("ML_MODEL_DIR", str(good))
    assert resolve_model_dir() == good
    monkeypatch.setenv("ML_MODEL_DIR", str(tmp_path / "nope"))
    assert resolve_model_dir() is None
    monkeypatch.delenv("ML_MODEL_DIR")
    assert resolve_model_dir(str(good)) == good


def test_load_bundle_unavailable_when_nothing_resolves(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_MODEL_DIR", str(tmp_path / "absent"))
    with pytest.raises(BundleUnavailable):
        load_bundle()


def test_load_bundle_rejects_incomplete_dir(tmp_path, monkeypatch):
    d = tmp_path / "bundle"
    d.mkdir()
    (d / "labels.json").write_text("{}")  # no tokenizer/session/checkpoint
    monkeypatch.setenv("ML_MODEL_DIR", str(d))
    with pytest.raises(BundleLoadError):
        load_bundle()
