"""OOD probe + routing-threshold artifact tests (#18 / #25)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mailroom_ml.calibration import (
    load_routing_thresholds,
    routing_thresholds_from_sweep,
    write_routing_thresholds,
)
from mailroom_ml.config import (
    BERT_INTAKE_MIN_CONFIDENCE,
    OOD_PROBE_FILENAME,
    ROUTING_THRESHOLDS_FILENAME,
)
from mailroom_ml.inference import classify_document
from mailroom_ml.ood import (
    energy_score,
    fit_ood_probe,
    load_ood_probe,
    score_ood,
    write_ood_probe,
)
from tests.test_inference import _logits_like, _stub_bundle


def test_energy_score_higher_for_flat_logits():
    peaked = energy_score(np.array([[10.0, 0.0, 0.0]]))
    flat = energy_score(np.array([[0.0, 0.0, 0.0]]))
    assert flat[0] > peaked[0]


def test_fit_and_score_ood_probe():
    rng = np.random.RandomState(0)
    in_dist = rng.normal(loc=[6.0, 0.0, 0.0], scale=0.2, size=(80, 3))
    probe = fit_ood_probe(in_dist, percentile=5.0)
    assert probe["method"] == "energy"
    assert probe["direction"] == "above"
    score, flag, status = score_ood(in_dist[:5], probe)
    assert status == "ok"
    assert flag is False
    ood = np.zeros((3, 3))
    score2, flag2, _ = score_ood(ood, probe)
    assert flag2 is True
    assert score2 > probe["threshold"]
    assert score_ood(ood, None) == (None, None, "absent")


def test_ood_probe_roundtrip(tmp_path: Path):
    probe = fit_ood_probe(np.array([[5.0, 0.0], [4.5, 0.2]]))
    dest = write_ood_probe(tmp_path, probe)
    assert dest.name == OOD_PROBE_FILENAME
    loaded = load_ood_probe(tmp_path)
    assert loaded["threshold"] == probe["threshold"]


def test_classify_document_fails_closed_on_ood():
    dt = [10.0, 0.0, 0.0, 0.0, 0.0, -5.0]
    corr = [0.0, 0.0, 9.0, 0.0]
    b = _stub_bundle(
        lambda ids, mask: _logits_like(
            "x", {"doc_type": dt, "correspondence": corr},
            n_windows=ids.shape[0]),
        support={"correspondence": {"notice": 20}},
    )
    b.ood_probe = {"method": "energy", "threshold": -1000.0,
                   "temperature": 1.0, "direction": "above"}
    res = classify_document(b, "Notice", "A notice demanding payment.",
                            window_texts=["notice text"])
    assert res["ood_flag"] is True
    assert res["route"] == "llm"
    assert res["reason"] == "ood"
    assert res["ood_probe_status"] == "ok"


def test_routing_thresholds_from_sweep_and_overlay(tmp_path: Path):
    sweep = {
        "recommended_threshold": 0.91,
        "refused": False,
        "budget_met": True,
        "selective_risk": 0.01,
        "coverage": 0.4,
        "n_at_pick": 40,
    }
    payload = routing_thresholds_from_sweep(sweep)
    assert payload["source"] == "selective_risk_sweep"
    assert payload["BERT_INTAKE_MIN_CONFIDENCE"] == 0.91
    assert payload["ROUTE_DOC_CONFIDENCE"] == 0.91
    dest = write_routing_thresholds(tmp_path, payload)
    assert dest.name == ROUTING_THRESHOLDS_FILENAME
    loaded = load_routing_thresholds(tmp_path)
    assert loaded["ROUTE_DOC_CONFIDENCE"] == 0.91

    refused = routing_thresholds_from_sweep(
        {"refused": True, "reason": "no sidecar"})
    assert refused["source"] == "config_fallback"
    assert refused["BERT_INTAKE_MIN_CONFIDENCE"] == BERT_INTAKE_MIN_CONFIDENCE


def test_classify_document_uses_artifact_thresholds():
    dt = [2.0, 1.0, 1.0, 1.0, 1.0, 0.0]  # low confidence
    b = _stub_bundle(
        lambda ids, mask: _logits_like(
            "x", {"doc_type": dt, "contract": [9.0, 0.0]},
            n_windows=ids.shape[0]),
        support={"contract": {"service": 20}},
    )
    # default config 0.97 should fail
    res = classify_document(b, "T", "body", window_texts=["w"])
    assert res["route"] == "llm"
    # overlay a very low threshold -> fast path if other gates hold
    b.routing_thresholds = {
        "ROUTE_DOC_CONFIDENCE": 0.01,
        "ROUTE_SUBCLASS_CONFIDENCE": 0.01,
        "ROUTE_WINDOW_AGREEMENT": 0.01,
        "ROUTE_MARGIN": 0.0,
    }
    res2 = classify_document(b, "T", "body", window_texts=["w"])
    assert res2["route"] == "fast_path"
