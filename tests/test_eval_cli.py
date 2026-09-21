"""Eval CLI tests (plan §11, issues #92 M7): CLI surface + sampling math.

Hermetic: argparse surface and the stratified sampler only — no model, no
network, no stage tree.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mailroom_ml.calibration import ece, ece_from_conf
from training.eval_modernbert import build_parser, stratified_sample


def test_cli_surface_accepts_documented_flags():
    ns = build_parser().parse_args([
        "--checkpoint", "artifacts/onnx/model", "--stage", "/tmp/stage",
        "--subset", "test", "--sample", "50", "--seed", "42",
        "--max-length", "4096", "--selective-risk", "--json",
    ])
    assert ns.checkpoint == "artifacts/onnx/model"
    assert ns.stage == Path("/tmp/stage")
    assert ns.subset == "test"
    assert ns.sample == 50
    assert ns.seed == 42
    assert ns.max_length == 4096
    assert ns.selective_risk is True
    assert ns.as_json is True


def test_cli_defaults_match_plan_surface():
    ns = build_parser().parse_args([])
    assert ns.subset == "test"
    assert ns.sample == 50
    assert ns.seed == 42
    assert ns.checkpoint == ""
    assert ns.selective_risk is False
    assert ns.as_json is False


def test_stratified_sample_deterministic_and_bounded():
    filenames = [f"f{i:02d}" for i in range(30)]
    strata = ["contract"] * 10 + ["correspondence"] * 10 + ["insurance_claim"] * 10
    a = stratified_sample(filenames, strata, 4, seed=42)
    b = stratified_sample(filenames, strata, 4, seed=42)
    assert a == b  # deterministic
    assert len(a) == 12  # min(4, 10) per stratum -> 12 total
    from collections import Counter

    picked = Counter(fn[:1] for fn in a)  # f<digit> -> stratum group
    assert picked["f0"] <= 4 and picked["f1"] <= 4 and picked["f2"] <= 4
    # order preserved relative to the input
    pos = {fn: i for i, fn in enumerate(filenames)}
    assert [pos[fn] for fn in a] == sorted(pos[fn] for fn in a)


def test_stratified_sample_seed_changes_pick():
    filenames = [f"f{i:02d}" for i in range(10)]
    strata = ["contract"] * 10
    assert stratified_sample(filenames, strata, 3, seed=1) != \
        stratified_sample(filenames, strata, 3, seed=2)


def test_stratified_sample_above_stratum_size_keeps_all():
    filenames = ["a", "b", "c"]
    strata = ["x"] * 3
    assert stratified_sample(filenames, strata, 100, seed=0) == filenames


def test_ece_from_conf_matches_logit_ece():
    """The eval CLI measures window calibration from merged probabilities —
    ece_from_conf(conf, correct) must agree with ece() on the logits that
    produced them (the old path softmaxed confidences a second time and
    crashed with an AxisError on every real run)."""
    rng = np.random.RandomState(7)
    logits = rng.normal(size=(400, 5))
    labels = rng.randint(0, 5, size=400)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(int)
    assert ece_from_conf(conf, correct) == pytest.approx(ece(logits, labels))
    # 1-D input must not crash (regression: the eval CLI's exact call shape)
    assert 0.0 <= ece_from_conf(conf, correct) <= 1.0


# ---------------------------------------------------------------------------
# #104 cohort split + per-head ECE refusal (offline stub bundle)
# ---------------------------------------------------------------------------

DOC_TYPES = ("contract", "merger_agreement", "corporate_record",
             "correspondence", "insurance_claim", "unknown")


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


def _stub_bundle(predict_fn, *, exclusion_policy=None, head_exclusions=None):
    from mailroom_ml.inference import ModelBundle

    maps = {
        "doc_type": {"labels": list(DOC_TYPES),
                     "label2id": {k: i for i, k in enumerate(DOC_TYPES)},
                     "id2label": {str(i): k for i, k in enumerate(DOC_TYPES)}},
        "correspondence": {"labels": ["email", "letter", "notice",
                                      "attorney_demand"],
                           "label2id": {"email": 0, "letter": 1, "notice": 2,
                                        "attorney_demand": 3},
                           "id2label": {"0": "email", "1": "letter",
                                        "2": "notice", "3": "attorney_demand"}},
    }
    return ModelBundle(
        model_dir=Path("/stub"), maps=maps, temperatures={},
        support_counts={}, tokenizer=_FakeTok({"<s>": 1, "</s>": 2}),
        pad_id=0, model_kind="stub", predict_fn=predict_fn,
        head_exclusions=head_exclusions or {},
        exclusion_policy=exclusion_policy,
    )


def _logits_like(head_vecs: dict[str, list[float]], n_windows: int):
    return {h: np.tile(np.array(v, dtype=np.float32), (n_windows, 1))
            for h, v in head_vecs.items()}


def _confident_predict(ids, mask):
    dt = [0.0, 0.0, 0.0, 10.0, 0.0, -6.0]  # correspondence wins
    corr = [0.0, 0.0, 10.0, 0.0]           # notice wins
    return _logits_like({"doc_type": dt, "correspondence": corr},
                        ids.shape[0])


def _eval_docs():
    import pandas as pd

    return pd.DataFrame([
        {"filename": "a.txt", "title": "T", "doc_text": "single",
         "doc_type": "correspondence", "subclass": "notice"},
        {"filename": "b.txt", "title": "T", "doc_text": "single",
         "doc_type": "correspondence", "subclass": "notice"},
        {"filename": "c.txt", "title": "T", "doc_text": "multi",
         "doc_type": "correspondence", "subclass": "notice"},
    ])


def test_evaluate_documents_reports_cohorts_separately(monkeypatch):
    """#104: single-window vs multi-window cohorts scored separately; the
    single-window cohort's agreement is trivially 1.0 and must not be
    conflated with the multi-window cohort's."""
    import training.eval_modernbert as ev

    def _fake_window(title, text, max_tokens):
        return ["w1", "w2"] if text == "multi" else ["w1"]

    monkeypatch.setattr(ev, "window_document", _fake_window)
    report = ev.evaluate_documents(
        _stub_bundle(_confident_predict), _eval_docs(),
        sample=0, seed=42, max_length=8192)
    cohorts = report["cohorts"]
    assert cohorts["single-window"]["n_docs"] == 2
    assert cohorts["multi-window"]["n_docs"] == 1
    assert cohorts["single-window"]["doc_type_accuracy"] == 1.0
    assert cohorts["multi-window"]["doc_type_accuracy"] == 1.0
    assert cohorts["single-window"]["mean_agreement"] == 1.0
    assert "window_ece" in cohorts["single-window"]
    assert report["head_ece"] == {}  # no sidecar -> empty


def test_sweep_refused_without_ece_sidecar(monkeypatch):
    """#104: no per-head ECE sidecar (pre-#107 artifact) -> the threshold
    sweep refuses instead of recommending from unaudited calibration."""
    import training.eval_modernbert as ev

    monkeypatch.setattr(ev, "window_document",
                        lambda title, text, max_tokens: ["w1"])
    report = ev.evaluate_documents(
        _stub_bundle(_confident_predict), _eval_docs(),
        sample=0, seed=42, max_length=8192, selective_risk=True)
    sr = report["selective_risk"]
    assert sr["refused"] is True
    assert "sidecar" in sr["reason"]


def test_sweep_refused_when_head_ece_over_threshold(monkeypatch):
    """#104: doc_type ECE >= HEAD_ECE_EXCLUSION_THRESHOLD (uncalibratable)
    -> refused, never a threshold pass."""
    import training.eval_modernbert as ev

    policy = {"budget": 0.05, "excluded": {
        "doc_type": {"excluded": False, "ece_calibrated": 0.161},
        "contract": {"excluded": True, "ece_calibrated": 0.09}}}
    monkeypatch.setattr(ev, "window_document",
                        lambda title, text, max_tokens: ["w1"])
    report = ev.evaluate_documents(
        _stub_bundle(_confident_predict, exclusion_policy=policy),
        _eval_docs(), sample=0, seed=42, max_length=8192,
        selective_risk=True)
    sr = report["selective_risk"]
    assert sr["refused"] is True
    assert "0.161" in sr["reason"]
    # the sidecar surfaces in the report regardless
    assert report["head_ece"]["doc_type"] == 0.161


def test_sweep_runs_with_clean_sidecar(monkeypatch):
    """#104: a clean per-head ECE sidecar lets the min-n sweep run (and
    report insufficient data honestly when n < min_n)."""
    import training.eval_modernbert as ev

    policy = {"budget": 0.05, "excluded": {
        "doc_type": {"excluded": False, "ece_calibrated": 0.02}}}
    monkeypatch.setattr(ev, "window_document",
                        lambda title, text, max_tokens: ["w1"])
    report = ev.evaluate_documents(
        _stub_bundle(_confident_predict, exclusion_policy=policy),
        _eval_docs(), sample=0, seed=42, max_length=8192,
        selective_risk=True)
    sr = report["selective_risk"]
    assert sr["refused"] is False
    assert sr["min_n"] == 30
    # 3 docs -> 3 windows -> n < min_n at every threshold -> honest refusal
    assert sr["insufficient_data"] is True
    assert sr["budget_met"] is False
