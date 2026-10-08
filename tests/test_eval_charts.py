"""Per-document-type SVG charts from eval JSON (no torch)."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mailroom_ml.viz import eval_charts as ec
from mailroom_ml.viz import svgcharts as sc

FIXTURE = Path(__file__).parent / "fixtures" / "eval_report_minimal.json"
SVG = "{http://www.w3.org/2000/svg}"


def _report(n_contract=40, collapsed=True):
    """Synthetic eval JSON: contracts collapse onto one subclass, others are fine."""
    per_doc = []
    for i in range(n_contract):
        gt = ["service", "license", "supply", "hosting"][i % 4]
        per_doc.append({"gt_doc_type": "contract", "pred_doc_type": "contract", "dt_correct": True,
                        "gt_subclass": gt, "pred_subclass": "license" if collapsed else gt,
                        "sc_correct": (gt == "license") if collapsed else True,
                        "n_windows": 1 + i % 12, "ood_flag": i % 7 == 0, "fast_path": i % 3 == 0})
    for i in range(30):
        ok = i % 10 != 0
        per_doc.append({"gt_doc_type": "correspondence", "pred_doc_type": "correspondence" if ok else "contract",
                        "dt_correct": ok, "gt_subclass": ["email", "letter"][i % 2], "pred_subclass": ["email", "letter"][i % 2],
                        "sc_correct": True, "n_windows": 1 + i % 3, "ood_flag": False, "fast_path": True})
    conf = {"contract->contract": n_contract, "correspondence->correspondence": 27, "correspondence->contract": 3}
    rows = [{"threshold": t / 100, "accuracy": 0.9 + t / 1000, "coverage": 1 - t / 200, "n": 70,
             "selective_risk": 0.1 - t / 1000, "wilson_lower_acc": 0.85 + t / 1000} for t in range(60, 100, 5)]
    return {"n_docs": len(per_doc), "per_doc": per_doc, "per_stratum_confusion": conf,
            "selective_risk": {"rows": rows, "recommended_threshold": 0.9, "error_budget": 0.02},
            "head_ece": {"doc_type": 0.02, "contract": 0.03, "correspondence": 0.07},
            "window_calibration": {"ece": 0.026}}


def _texts(svg: str) -> list[str]:
    return [t.text for t in ET.fromstring(svg).iter(f"{SVG}text")]


def test_confusion_and_recall_from_per_stratum_confusion():
    rep = _report()
    assert ec.confusion(rep)["correspondence"] == {"correspondence": 27, "contract": 3}
    rec = ec.recall_by_class(rep)
    assert rec["contract"] == 1.0
    assert rec["correspondence"] == pytest.approx(0.9)
    assert rec["insurance_claim"] is None  # no support -> no bar


def test_subclass_stats_expose_collapse():
    s = ec.subclass_stats(_report())["contract"]
    assert s["n"] == 40 and s["correct"] == 10
    assert s["top_pred"] == ("license", 40)  # every contract predicted one subclass
    assert s["top_true"][1] == 10


def test_render_writes_every_chart_as_valid_svg():
    charts = ec.render([("Run 3", _report(collapsed=False)), ("Arm B", _report())])
    assert set(charts) >= {"doc_type_recall.svg", "confusion.svg", "subclass_collapse.svg", "selective_risk.svg",
                           "head_ece.svg", "subclass_contract.svg", "subclass_correspondence.svg",
                           "accuracy_by_windows.svg", "ale_windows.svg"}
    for svg in charts.values():
        assert ET.fromstring(svg).tag == f"{SVG}svg"
    assert "top prediction 'license' takes 40/40" in " ".join(_texts(charts["subclass_contract.svg"]))
    assert "Pick 0.90" in " ".join(_texts(charts["selective_risk.svg"]))
    legend = _texts(charts["doc_type_recall.svg"])
    assert "Run 3" in legend and "Arm B" in legend


def test_selective_risk_pick_is_matched_by_tolerance_not_float_equality():
    """The recommended threshold round-trips through JSON/arithmetic: a 1-ulp
    difference must still find its row (not report 'no threshold met')."""
    rep = _report()
    pick = 0.6 + 0.3
    assert pick != 0.9 and abs(pick - 0.9) < 1e-12
    rep["selective_risk"]["recommended_threshold"] = pick
    charts = ec.render([("Run 3", rep)])
    sub = " ".join(_texts(charts["selective_risk.svg"]))
    assert "Pick 0.90" in sub
    assert "No threshold met" not in sub


def test_lines_degenerate_explicit_y_range_does_not_divide_by_zero():
    """y_range=(v, v) used to ZeroDivisionError in the Y() scale."""
    svg = sc.lines("flat", [{"name": "a", "color": sc.BLUE, "x": [0, 1, 2], "y": [1.0, 1.0, 1.0]}],
                   y_range=(1.0, 1.0))
    assert ET.fromstring(svg).tag == f"{SVG}svg"


def test_render_is_deterministic():
    a = ec.render([("Arm B", _report())])
    b = ec.render([("Arm B", json.loads(json.dumps(_report())))])
    assert a == b


def test_eval_without_per_doc_still_gets_aggregate_charts():
    rep = _report()
    rep.pop("per_doc")
    charts = ec.render([("Run 3", rep)])
    assert "confusion.svg" in charts and "doc_type_recall.svg" in charts
    assert not any(k.startswith("subclass_") or k == "ale_windows.svg" for k in charts)


def test_minimal_fixture_renders():
    rep = json.loads(FIXTURE.read_text(encoding="utf-8"))
    charts = ec.render([("fixture", rep)])
    assert "doc_type_recall.svg" in charts


def test_cli_writes_gallery(tmp_path):
    src = tmp_path / "eval_armB.json"
    src.write_text(json.dumps(_report()))
    assert ec.main([str(src), "--out", str(tmp_path / "charts")]) == 0
    md = (tmp_path / "charts" / "README.md").read_text()
    assert "![Document-type recall by class](doc_type_recall.svg)" in md
    assert "armB" in md  # default label is the file stem minus "eval_"
    for line in md.splitlines():
        if line.startswith("!["):
            assert (tmp_path / "charts" / line.rsplit("(", 1)[1].rstrip(")")).exists()


def test_cli_rejects_label_count_mismatch(tmp_path):
    src = tmp_path / "eval_x.json"
    src.write_text(json.dumps(_report()))
    with pytest.raises(SystemExit):
        ec.main([str(src), str(src), "--label", "only-one", "--out", str(tmp_path)])


ROOT = Path(__file__).resolve().parent.parent
GALLERIES = {
    "m9a-local-20260927-014429": [("Run 3 (Sep 21)", "eval_run3_20260921.json"),
                                  ("Arm A (Sep 27)", "eval_m9a-local-gpu1-armA-1ep-20260927-025236.json"),
                                  ("Arm B (Sep 27)", "eval_m9a-local-20260927-014429.json")],
    "m9a-local-gpu1-armA-1ep-20260927-025236": [("Run 3 (Sep 21)", "eval_run3_20260921.json"),
                                                ("Arm A (Sep 27)", "eval_m9a-local-gpu1-armA-1ep-20260927-025236.json")],
}


@pytest.mark.parametrize("tag", sorted(GALLERIES))
def test_committed_galleries_match_their_eval_json(tag):
    """reports/charts/<tag> is regenerated from the committed eval JSON, byte for byte."""
    runs = [(lab, json.loads((ROOT / "reports" / "json" / f).read_text())) for lab, f in GALLERIES[tag]]
    out = ROOT / "reports" / "charts" / tag
    for name, svg in ec.render(runs).items():
        assert (out / name).read_text() == svg, f"{tag}/{name} is stale: re-run write_eval_report.py charts"
