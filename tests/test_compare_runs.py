"""Hermetic tests for training/compare_runs.py (#17). No torch."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.compare_runs import (
    build_parser,
    compare_reports,
    format_markdown,
    load_eval_json,
    paired_bootstrap,
)


def _eval(n: int = 4, acc_shift: int = 0, head_f1: float = 0.4) -> dict:
    docs = []
    for i in range(n):
        ok = (i + acc_shift) % 2 == 0
        docs.append({
            "filename": f"d{i}.txt",
            "gt_doc_type": "contract",
            "pred_doc_type": "contract" if ok else "correspondence",
            "dt_correct": ok,
            "gt_subclass": "service",
            "pred_subclass": "service" if ok else "license",
            "sc_correct": ok,
            "n_windows": 1 if i < n // 2 else 3,
            "agreement": 1.0,
            "fast_path": ok,
            "ood_flag": False,
        })
    return {
        "n_docs": n,
        "doc_type_accuracy": sum(d["dt_correct"] for d in docs) / n,
        "fast_path_rate": sum(d["fast_path"] for d in docs) / n,
        "window_calibration": {"ece": 0.02, "band_ece": 0.01, "n_windows": n},
        "selective_risk": {"recommended_threshold": 0.9, "refused": False},
        "per_head": {"contract": {"macro_f1": head_f1, "support": {"service": n}}},
        "ood": {"rate": 0.0, "n_flagged": 0, "n_scored": n, "probe_status": "absent"},
        "per_doc": docs,
        "cohorts": {
            "single-window": {"doc_type_accuracy": 1.0, "window_ece": 0.01},
            "multi-window": {"doc_type_accuracy": 0.5, "window_ece": 0.03},
        },
    }


def test_paired_bootstrap_zero_diff_includes_zero():
    a = [1.0, 0.0, 1.0, 0.0]
    out = paired_bootstrap(a, a, n_resamples=200, seed=42)
    assert out["n_paired"] == 4
    assert out["mean_diff"] == pytest.approx(0.0)
    assert out["ci_low"] <= 0.0 <= out["ci_high"]


def test_compare_reports_pairs_on_filename():
    a = _eval(acc_shift=0, head_f1=0.5)
    b = _eval(acc_shift=1, head_f1=0.2)
    cmp = compare_reports(a, b, n_resamples=200, seed=1)
    assert cmp["n_paired"] == 4
    assert cmp["pairing"] == "document_id"
    assert cmp["metrics"]["doc_type_accuracy"]["paired"]["n_paired"] == 4
    assert cmp["per_head"]["contract"]["delta"] == pytest.approx(0.3)


def test_compare_reports_no_per_doc_is_unpaired():
    a = _eval()
    b = _eval()
    a.pop("per_doc")
    b.pop("per_doc")
    cmp = compare_reports(a, b, n_resamples=50)
    assert cmp["n_paired"] == 0
    assert cmp["pairing"] == "none"
    assert cmp["metrics"]["doc_type_accuracy"]["paired"] is None


def test_cli_and_markdown_and_json_roundtrip(tmp_path: Path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_eval()), encoding="utf-8")
    b.write_text(json.dumps(_eval(acc_shift=1)), encoding="utf-8")
    loaded = load_eval_json(a)
    assert loaded["n_docs"] == 4
    ns = build_parser().parse_args([
        "--a", str(a), "--b", str(b), "--json", "--resamples", "100",
    ])
    assert ns.resamples == 100
    cmp = compare_reports(load_eval_json(a), load_eval_json(b), n_resamples=100)
    md = format_markdown(cmp, label_a="clf", label_b="llm")
    assert "compare_runs" in md
    assert "doc_type_accuracy" in md
    assert "contract" in md
