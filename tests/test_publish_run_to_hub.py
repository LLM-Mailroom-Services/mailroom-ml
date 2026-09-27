"""Hermetic tests for Hub publish planning (no network, no token)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from training.publish_run_to_hub import (
    METRICS_BEGIN,
    METRICS_END,
    build_metrics_markdown,
    gates_all_met,
    patch_readme_metrics,
)


def test_gates_all_met_from_fixture():
    report = {
        "doc_type_accuracy": 0.90,
        "window_calibration": {"ece": 0.04},
        "per_head": {
            "contract": {"macro_f1": 0.21},
            "correspondence": {"macro_f1": 0.26},
        },
    }
    assert gates_all_met(report) is True
    report["per_head"]["contract"]["macro_f1"] = 0.19
    assert gates_all_met(report) is False


def test_metrics_markdown_includes_gate_table():
    md = build_metrics_markdown(
        release_tag="m9a-local-test",
        released_at="2026-09-27",
        eval_artifact_name="eval_report_m9a-local-test.json",
        report={
            "doc_type_accuracy": 0.8947,
            "subclass_accuracy_conditional": 0.526,
            "window_calibration": {"ece": 0.0203, "band_ece": 0.05},
            "per_head": {
                "contract": {
                    "macro_f1": 0.0395,
                    "support": 12,
                    "ece_calibrated": 0.0443,
                },
            },
        },
        summary={"checkpoint_selection": {"epoch": 1, "gate_met": True}},
    )
    assert "M9a #112 gates" in md
    assert "contract test macro-F1" in md
    assert "eval_report_m9a-local-test.json" in md


def test_patch_readme_replaces_marked_section():
    old = f"intro\n{METRICS_BEGIN}\nold\n{METRICS_END}\ntail"
    new_block = "## Release `x`"
    out = patch_readme_metrics(old, new_block)
    assert "intro" in out and "tail" in out
    assert "old" not in out
    assert new_block in out


def test_cli_dry_run_missing_checkpoint(tmp_path: Path):
    rc = subprocess.call(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "training/publish_run_to_hub.py"),
            "--dry-run",
            "--checkpoint",
            str(tmp_path / "nope"),
            "--eval-json",
            str(tmp_path / "eval.json"),
        ],
    )
    assert rc == 2


def test_cli_dry_run_prints_plan(tmp_path: Path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "summary.json").write_text(json.dumps({"run_id": "rid-1"}), encoding="utf-8")
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "doc_type_accuracy": 0.9,
                "window_calibration": {"ece": 0.01},
                "per_head": {
                    "contract": {"macro_f1": 0.3},
                    "correspondence": {"macro_f1": 0.3},
                },
            }
        ),
        encoding="utf-8",
    )
    out = subprocess.check_output(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "training/publish_run_to_hub.py"),
            "--dry-run",
            "--checkpoint",
            str(ckpt),
            "--eval-json",
            str(eval_path),
            "--release-tag",
            "t1",
        ],
        text=True,
    )
    assert "Hub publish plan" in out
    assert "Lucius-Morningstar/mailroom-modernbert-classifier" in out
