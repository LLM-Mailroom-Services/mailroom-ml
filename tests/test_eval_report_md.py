"""Pin artifact-driven eval markdown (no torch)."""
from __future__ import annotations

import json
from pathlib import Path

from mailroom_ml.m9a_gates import gates_all_met, m9a_gate_rows
from mailroom_ml.eval_report_md import ReportContext, render_test_eval_report

FIXTURE = Path(__file__).parent / "fixtures" / "eval_report_minimal.json"


def test_m9a_gates_met_on_minimal_fixture():
    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert gates_all_met(report) is True
    rows = m9a_gate_rows(report)
    assert len(rows) == 4
    assert rows[0][1] == 0.21


def test_render_test_eval_contains_headline_metrics():
    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ctx = ReportContext(run_tag="fixture-run", arm="B")
    md = render_test_eval_report(report, ctx=ctx, summary=None)
    assert "## Headline test metrics (eval harness)" in md
    assert "0.9000" in md or "0.90" in md
    assert "doc_type_accuracy" in md
    assert "## M9a #112 gates (held-out test)" in md
    assert "## Analyst insights & findings" in md
    assert "## Reproduce" in md
    assert "## Artifacts" in md
    assert "PASS" in md or "FAIL" in md


def test_render_uses_eval_json_doc_type_not_hand_value():
    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report["doc_type_accuracy"] = 0.1111
    report["recorded_gates"]["P0_doc_type"]["actual"] = 0.1111
    ctx = ReportContext(run_tag="t")
    md = render_test_eval_report(report, ctx=ctx)
    assert "**0.1111**" in md
