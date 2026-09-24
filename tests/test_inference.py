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
    project_subclass,
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


@pytest.mark.train
def test_load_tokenizer_neutralizes_baked_padding_and_truncation(tmp_path):
    """The trainer's tokenizer.json bakes padding=longest + truncation=8192;
    the serving loader must neutralize BOTH so every window does not pad to
    the full 8,192-token tensor (run-3 regression: minutes-per-doc on CPU
    fp32) and so encode_inputs sees true lengths (no-truncation doctrine)."""
    pytest.importorskip("tokenizers")
    import tokenizers as tkz

    vocab = {v: i for i, v in enumerate(
        ["<s>", "</s>", "<unk>", "<pad>", "hello", "world", "hi", "x"])}
    tok = tkz.Tokenizer(tkz.models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = tkz.pre_tokenizers.Whitespace()
    tok.post_processor = tkz.processors.TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 1)])
    tok.enable_truncation(8192)
    tok.enable_padding(pad_id=3, pad_token="<pad>")
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))

    from mailroom_ml.inference import _load_tokenizer, encode_inputs

    loaded = _load_tokenizer(tmp_path)
    b = _stub_bundle(None, tokenizer=loaded, pad_id=3)

    # Regression: a short doc encoded for serving must keep its TRUE length
    # (4 = <s> + 2 content + </s>) — the baked padding=longest would force
    # this to an 8192-wide tensor (fp32 CPU: minutes per document).
    ids, mask = encode_inputs(b, ["hello world"])
    assert ids.shape == (1, 4), f"padding not neutralized: {ids.shape}"
    assert mask.tolist() == [[1, 1, 1, 1]]

    # No-truncation doctrine: an over-long input must RAISE (route LLM),
    # never be silently cut by the baked truncation=8192.
    with pytest.raises(ValueError):
        encode_inputs(b, [" ".join(["x"] * 9000)], max_length=8192)


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


def test_runner_up_never_equals_winner_when_second_best_exists():
    """Plurality winner can differ from mean-prob ranking (#124)."""
    corr_win = [0.0, 0.0, 0.0, 10.0, 0.0, -5.0]
    ins_spike = [0.0, 0.0, 0.0, 0.0, 50.0, -5.0]
    sequences = [corr_win, corr_win, ins_spike]

    def predict(ids, mask):
        row = sequences.pop(0)
        n = ids.shape[0]
        return {
            "doc_type": np.tile(np.array(row, dtype=np.float32), (n, 1)),
            "correspondence": np.tile(np.array([0.0, 0.0, 9.0, 0.0],
                                               dtype=np.float32), (n, 1)),
        }

    b = _stub_bundle(predict)
    res = classify_windows(b, ["w1", "w2", "w3"])
    assert res["doc_type"] == "correspondence"
    assert res["runner_up"] != "correspondence"
    assert res["margin"] > 0.0


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


# ---------------------------------------------------------------------------
# classify_document_default — the M6a lane seam (bundle-less entrypoint)
# ---------------------------------------------------------------------------


def test_default_seam_happy_path_uses_stem_title(monkeypatch):
    from mailroom_ml.inference import (
        classify_document_default,
        clear_default_bundle_cache,
    )

    clear_default_bundle_cache()
    b = _confident_doc_bundle()
    monkeypatch.setattr("mailroom_ml.inference.load_bundle", lambda *a, **k: b)
    res = classify_document_default(
        "A notice demanding payment.",
        filename="/inbox/notice_file.pdf",
        window_texts=["notice text"],
    )
    assert res["status"] == "ok"
    assert res["doc_type"] == "correspondence"
    assert res["subclass"] == "notice"
    assert res["route"] == "fast_path"
    assert res["reason"] == "fast_path"
    clear_default_bundle_cache()


def test_default_seam_missing_bundle_fails_open_no_raise(monkeypatch):
    from mailroom_ml.inference import (
        BundleUnavailable,
        classify_document_default,
        clear_default_bundle_cache,
    )

    clear_default_bundle_cache()

    def _boom(*a, **k):
        raise BundleUnavailable("no bundle")

    monkeypatch.setattr("mailroom_ml.inference.load_bundle", _boom)
    res = classify_document_default("x", filename="y.txt")
    assert res["status"] == "failure"
    assert res["route"] == "llm"
    assert res["reason"] == "no_model"  # lane marker vocabulary
    clear_default_bundle_cache()

    from mailroom_ml.inference import BundleLoadError

    def _boom2(*a, **k):
        raise BundleLoadError("bundle missing labels.json")

    monkeypatch.setattr("mailroom_ml.inference.load_bundle", _boom2)
    res2 = classify_document_default("x", filename="y.txt")
    assert res2["reason"] == "bundle_missing"
    clear_default_bundle_cache()


def test_default_seam_caches_bundle(monkeypatch):
    from mailroom_ml.inference import (
        classify_document_default,
        clear_default_bundle_cache,
    )

    clear_default_bundle_cache()
    b = _confident_doc_bundle()
    calls = []

    def _counting(*a, **k):
        calls.append(1)
        return b

    monkeypatch.setattr("mailroom_ml.inference.load_bundle", _counting)
    classify_document_default("a", window_texts=["a"])
    classify_document_default("b", window_texts=["b"])
    assert len(calls) == 1  # second call reused the cached bundle
    clear_default_bundle_cache()


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
# #107 taxonomy-conformance projections + head-exclusion policy
# ---------------------------------------------------------------------------

def test_project_subclass_certificate_of_formation_to_other():
    """Canonical 11-key corporate_record surface vs the observed 10-key head
    (#67): certificate_of_formation projects to the head's `other`."""
    maps = {"corporate_record": {"labels": [
        "charter_amendment", "articles_of_incorporation", "officer_certificate",
        "indenture", "subsidiary_list", "rights_instrument",
        "board_resolution", "bylaws", "powers_of_attorney", "other"]}}
    assert project_subclass(maps, "corporate_record",
                            "certificate_of_formation") == "other"
    # in-vocab labels pass through untouched
    assert project_subclass(maps, "corporate_record", "bylaws") == "bylaws"
    assert project_subclass(maps, "corporate_record", None) is None


def test_project_subclass_insurance_unmapped_returns_none():
    """6-key insurance head with NO `other` (#68): any canonical subclass
    outside the head vocab is unmapped (None) — the caller routes LLM,
    never force-fits into a sibling class."""
    maps = {"insurance_claim": {"labels": [
        "carrier", "inpatient", "outpatient", "pde", "property", "auto"]}}
    assert project_subclass(maps, "insurance_claim", "dental_claim") is None
    assert project_subclass(maps, "insurance_claim", "carrier") == "carrier"
    # a head WITH `other` still falls back to it for unknown canonicals
    maps2 = {"contract": {"labels": ["service", "license", "other"]}}
    assert project_subclass(maps2, "contract", "transportation") == "other"


def test_head_exclusion_policy_routes_llm():
    """#107: a subclass head excluded by the artifact's calibration policy
    (calibrated ECE > budget) blocks the fast path for its doc_type."""
    b = _stub_bundle(
        lambda ids, mask: _logits_like(
            "x", {"doc_type": [0.0, 0.0, 0.0, 10.0, 0.0, -6.0],
                  "correspondence": [0.0, 0.0, 10.0, 0.0]}, 1),
        support={"correspondence": {"notice": 12}},
        head_exclusions={"correspondence": "ece_calibrated 0.12 > budget 0.05"},
        exclusion_policy={"budget": 0.05})
    res = classify_document(b, "T", "body", window_texts=["w"])
    assert res["doc_type"] == "correspondence"
    assert res["route"] == "llm"
    assert res["reason"] == "head_excluded"
    assert any("head_excluded:correspondence" in f
               for f in res["guard_failures"])
    assert res["quality"]["exclusion_policy"] == "present"


def test_unexcluded_head_still_fast_paths():
    """An artifact whose policy excludes OTHER heads leaves this one fast."""
    b = _stub_bundle(
        lambda ids, mask: _logits_like(
            "x", {"doc_type": [0.0, 0.0, 0.0, 10.0, 0.0, -6.0],
                  "correspondence": [0.0, 0.0, 10.0, 0.0]}, 1),
        support={"correspondence": {"notice": 12}},
        head_exclusions={"contract": "ece_calibrated 0.09 > budget 0.05"})
    res = classify_document(b, "T", "body", window_texts=["w"])
    assert res["route"] == "fast_path"
    assert res["quality"]["exclusion_policy"] == "absent"


def test_unmapped_subclass_routes_llm_never_force_fits():
    """Drift guard: a merged subclass the head cannot express (and with no
    `other` to project to) routes LLM — never a sibling-class force-fit."""
    # insurance head whose id2label drifted to carry a canonical label the
    # head vocab cannot express (the #66 parity failure mode)
    dt = [0.0, 0.0, 0.0, 0.0, 10.0, -6.0]  # insurance_claim wins
    maps = {
        "doc_type": HEADS["doc_type"],
        "insurance_claim": {
            "labels": ["carrier", "inpatient", "outpatient", "pde",
                       "property", "auto"],
            "label2id": {"carrier": 0, "inpatient": 1, "outpatient": 2,
                         "pde": 3, "property": 4, "auto": 5},
            "id2label": {"0": "carrier", "1": "inpatient", "2": "outpatient",
                         "3": "pde", "4": "property", "5": "dental_claim"},
        },
    }
    b = _stub_bundle(
        lambda ids, mask: _logits_like(
            "x", {"doc_type": dt,
                  "insurance_claim": [0.0, 0.0, 0.0, 0.0, 0.0, 10.0]}, 1),
        maps=maps, support={"insurance_claim": {"dental_claim": 12}})
    res = classify_document(b, "T", "body", window_texts=["w"])
    assert res["doc_type"] == "insurance_claim"
    assert res["route"] == "llm"
    assert res["reason"] == "subclass_unmapped"
    assert any("subclass_unmapped:insurance_claim/dental_claim" in f
               for f in res["guard_failures"])


# ---------------------------------------------------------------------------
# #103 catch-all guard + support-gate fixture matrix
# ---------------------------------------------------------------------------

CONTRACT_LABELS = ["service", "license", "transportation",
                   "non_compete_no_solicit", "attorney_demand",
                   "mixed_cash_stock_election", "other"]
# Verified counts from the live bundle's train_counts.json (#103 origin)
CONTRACT_SUPPORT = {
    "transportation": 0, "non_compete_no_solicit": 2, "attorney_demand": 2,
    "mixed_cash_stock_election": 13, "other": 5, "service": 40,
    "license": 30,
}


def _contract_bundle(subclass: str, support: dict[str, int]):
    """Contract-doc stub bundle whose subclass head votes ``subclass``."""
    dt = [10.0, 0.0, 0.0, 0.0, 0.0, -6.0]  # contract wins
    sc = [0.0] * len(CONTRACT_LABELS)
    sc[CONTRACT_LABELS.index(subclass)] = 10.0
    maps = {
        "doc_type": HEADS["doc_type"],
        "contract": {
            "labels": CONTRACT_LABELS,
            "label2id": {k: i for i, k in enumerate(CONTRACT_LABELS)},
            "id2label": {str(i): k for i, k in enumerate(CONTRACT_LABELS)},
        },
    }
    return _stub_bundle(
        lambda ids, mask: _logits_like(
            "x", {"doc_type": dt, "contract": sc}, 1),
        maps=maps, support={"contract": support})


@pytest.mark.parametrize("subclass, support, expected_route, reason", [
    # zero-row label in the head -> support gate routes LLM (#103 pin)
    ("transportation", CONTRACT_SUPPORT, "llm", "insufficient_support"),
    # sub-floor labels (2 < ROUTE_MIN_AUTHENTIC_SUPPORT=5) -> LLM
    ("non_compete_no_solicit", CONTRACT_SUPPORT, "llm", "insufficient_support"),
    ("attorney_demand", CONTRACT_SUPPORT, "llm", "insufficient_support"),
    # at/above floor + thresholds clear -> fast path
    ("mixed_cash_stock_election", CONTRACT_SUPPORT, "fast_path", "fast_path"),
    # catch-all at the tier floor (5) -> hard-excluded regardless of support
    ("other", CONTRACT_SUPPORT, "llm", "catchall_label"),
])
def test_support_gate_fixture_matrix(subclass, support, expected_route, reason):
    """#103: the support-gate + catch-all fixture matrix from the live
    bundle's verified counts — a future head/vocab change cannot silently
    flip any of these routes."""
    b = _contract_bundle(subclass, support)
    res = classify_document(b, "T", "body", window_texts=["w"])
    assert res["doc_type"] == "contract"
    assert res["subclass"] == subclass
    assert res["route"] == expected_route
    assert res["reason"] == reason
    if reason == "insufficient_support":
        assert any("authentic support" in f for f in res["guard_failures"])
    if reason == "catchall_label":
        assert any(f"catchall_label:contract/{subclass}" in f
                   for f in res["guard_failures"])


def test_catchall_excluded_even_with_high_support():
    """#103: `other` with support far above the floor still never fast-paths
    — the catch-all exclusion is support-independent."""
    b = _contract_bundle("other", {"other": 500})
    res = classify_document(b, "T", "body", window_texts=["w"])
    assert res["route"] == "llm"
    assert res["reason"] == "catchall_label"
    assert "catchall_label:contract/other" in res["guard_failures"]


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


@pytest.mark.train
def test_head_from_state_reconstructs_linear_and_mlp():
    """heads.pt is self-describing: linear heads (weight/bias) and MLP heads
    (0.*/3.* — run-3 --mlp-heads) must both load. The loader previously
    assumed linear-only (eval harness failed on run-3: KeyError 'weight')."""
    pytest.importorskip("torch")
    import torch

    from mailroom_ml.inference import _head_from_state

    hidden = 8
    lin = torch.nn.Linear(hidden, 3)
    m = _head_from_state(lin.state_dict(), hidden)
    assert list(m.state_dict().keys()) == ["weight", "bias"]
    assert m(torch.randn(2, hidden)).shape == (2, 3)

    mlp = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
        torch.nn.Dropout(0.0), torch.nn.Linear(hidden, 5))
    m2 = _head_from_state(mlp.state_dict(), hidden)
    assert list(m2.state_dict().keys()) == ["0.weight", "0.bias",
                                            "3.weight", "3.bias"]
    assert m2(torch.randn(2, hidden)).shape == (2, 5)
