"""Tracing-contract tests (plan §9 span surface, issues #85 M1/M3).

The spans are pure dicts — no tracer attached — so the contract surface is
fully testable: exact key sets, metadata identity fields, failure records.
"""
from __future__ import annotations

from mailroom_ml.config import INTAKE_HANDOFF_SCHEMA_VERSION
from mailroom_ml.tracing import (
    SPAN_INTAKE_BERT_PREP,
    SPAN_INTAKE_ML_TRIAGE,
    build_bert_prep_span,
    build_ml_triage_span,
    build_section_map,
    record_failure,
)


def test_bert_prep_span_contract_surface():
    s = build_bert_prep_span(
        filename="a.txt", doc_chars=1200, token_estimate=300, windows=1,
        model_id="modernbert-base", artifact_sha="abc123",
        dataset_revision="rev1", calibration_version="v3",
        label_schema_version="heads6", synthetic_policy_version="v1",
        window_agreement=0.9, route="fast_path", doc_type="correspondence",
        subclass="notice", confidence=0.97,
    )
    assert s["name"] == SPAN_INTAKE_BERT_PREP
    assert set(s) == {"name", "input", "output", "metadata", "quality"}
    assert set(s["input"]) == {"filename", "chars", "token_estimate", "windows"}
    assert set(s["output"]) == {"doc_type", "subclass", "confidence",
                                "route", "window_agreement"}
    assert set(s["metadata"]) == {"model_id", "artifact_sha",
                                  "dataset_revision", "calibration_version",
                                  "label_schema_version",
                                  "synthetic_policy_version"}
    assert s["metadata"]["artifact_sha"] == "abc123"
    assert s["quality"] == {}


def test_ml_triage_span_contract_surface():
    s = build_ml_triage_span(
        method="bert", routing_path="bert_intake",
        routing_reason="within_bert_ctx", doc_chars=900, token_estimate=220,
        windows=2, model_id="modernbert-base", artifact_sha="sha",
        dataset_revision="rev", calibration_version="v3",
        label_schema_version="v1", synthetic_policy_version="v1",
        window_agreement=0.8, doc_type="contract", subclass="service",
        confidence=0.95, route="fast_path",
    )
    assert s["name"] == SPAN_INTAKE_ML_TRIAGE
    assert s["metadata"]["schema_version"] == INTAKE_HANDOFF_SCHEMA_VERSION
    assert set(s["input"]) == {"chars", "token_estimate", "windows"}
    assert set(s["output"]) == {"doc_type", "subclass", "confidence",
                                "route", "window_agreement"}
    meta = s["metadata"]
    assert {"model_id", "artifact_sha", "dataset_revision",
            "calibration_version", "label_schema_version",
            "synthetic_policy_version"} <= set(meta)
    assert s["guard_failures"] == []


def test_record_failure_attaches_machine_readable_blob():
    span = build_ml_triage_span(method="bert", routing_path="bert_intake",
                                routing_reason="within_bert_ctx", doc_chars=1,
                                token_estimate=1, windows=1, model_id="m",
                                artifact_sha=None, dataset_revision=None,
                                calibration_version=None,
                                label_schema_version=None,
                                synthetic_policy_version=None,
                                window_agreement=None)
    try:
        raise RuntimeError("onnx session died")
    except RuntimeError as exc:
        broken = record_failure(span, exc, where="load_bundle")
    assert broken["failure"]["type"] == "RuntimeError"
    assert broken["failure"]["where"] == "load_bundle"
    assert "onnx" in broken["failure"]["message"]
    assert "traceback_point" in broken["failure"]
    # original span keys survive the wrap
    assert broken["name"] == SPAN_INTAKE_ML_TRIAGE


def test_section_map_empty_by_design_is_ok():
    res = build_section_map([], doc_chars=0)
    assert res["ok"] is True
    assert res["sections"] == []
    assert res["guard_failures"] == []


def test_section_map_validates_offsets_and_roles():
    doc_chars = 400
    good = [{"heading": "H1", "role": "notice", "start_offset": 0,
             "end_offset": 100},
            {"heading": "H2", "role": "body", "start_offset": 120,
             "end_offset": 300}]
    res = build_section_map(good, doc_chars,
                            roles_catalog=frozenset({"notice", "body"}))
    assert res["ok"] is True
    assert len(res["sections"]) == 2

    # out of bounds
    bad = [{"heading": "H", "role": "notice", "start_offset": 10,
            "end_offset": 500}]
    res = build_section_map(bad, doc_chars)
    assert res["ok"] is False
    assert any("out of bounds" in f for f in res["guard_failures"])

    # overlap
    over = [{"heading": "A", "role": "x", "start_offset": 0, "end_offset": 100},
            {"heading": "B", "role": "y", "start_offset": 90, "end_offset": 200}]
    res = build_section_map(over, doc_chars)
    assert res["ok"] is False
    assert any("overlaps" in f for f in res["guard_failures"])

    # unknown role against a catalog
    res = build_section_map(good, doc_chars,
                            roles_catalog=frozenset({"notice"}))
    assert res["ok"] is False
    assert any("not in catalog" in f for f in res["guard_failures"])
