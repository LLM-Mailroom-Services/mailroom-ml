"""Pin artifact-driven eval markdown (no torch)."""
from __future__ import annotations

import json
from pathlib import Path

from mailroom_ml.eval_report_md import ReportContext, render_test_eval_report
from mailroom_ml.m9a_gates import gates_all_met, m9a_gate_rows

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


def test_gates_nan_metric_is_missing_not_met():
    """NaN compares False against every threshold: it must fail the gate,
    and the boolean must agree with the per-row status."""
    from mailroom_ml.m9a_gates import format_gates_cli, gate_status

    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report["window_calibration"]["ece"] = float("nan")
    rows = m9a_gate_rows(report)
    assert rows[3][1] is None
    assert gate_status(rows[3][1], rows[3][2], rows[3][3]) == "MISSING"
    assert gates_all_met(report) is False
    assert format_gates_cli(report).endswith("FAIL")


def _assert_tables_well_formed(md: str) -> None:
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and (i == 0 or not lines[i - 1].startswith("|")):
            # every table opens with a header row then a separator row
            assert i + 1 < len(lines) and lines[i + 1].replace(" ", "").startswith(
                "|---"), f"table without header/separator: {line}"
            ncols = line.count("|")
            j = i + 1
            while j < len(lines) and lines[j].startswith("|"):
                assert lines[j].count("|") == ncols, f"ragged row: {lines[j]}"
                j += 1


def test_rendered_tables_are_well_formed_and_paths_current():
    from mailroom_ml.eval_report_md import (
        render_compare_report,
        render_heldout_plus_report,
    )

    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    md = render_test_eval_report(report, ctx=ReportContext(run_tag="t"))
    assert "training/eval/eval_modernbert.py" in md
    assert "training/eval_modernbert.py" not in md
    assert "training/write_eval_report.py" not in md
    # thresholds are fit on validation, never on the test eval (plan D11)
    assert "--subset validation" in md
    _assert_tables_well_formed(md)
    hp = render_heldout_plus_report(
        report, ctx=ReportContext(run_tag="t"), plus_filenames=set())
    assert "training/eval/write_eval_report.py" in hp
    _assert_tables_well_formed(hp)
    cmp = {"n_docs_a": 1, "n_docs_b": 1, "n_paired": 0, "metrics": {},
           "per_head": {}, "paired": {}, "pairing": "none"}
    cr = render_compare_report(cmp, label_a="A", label_b="B",
                               path_a="a.json", path_b="b.json")
    assert "training/eval/compare_runs.py" in cr
    _assert_tables_well_formed(cr)
