"""Hermetic tests for Hub publish planning (no network, no token)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from mailroom_ml.m9a_gates import gates_all_met

# ``post-train`` is a hyphenated directory (not importable as a package), so
# the publish script is loaded by file path.
_PUBLISH_PATH = (
    Path(__file__).resolve().parents[1]
    / "training" / "train" / "post-train" / "publish_run_to_hub.py"
)
_spec = importlib.util.spec_from_file_location("publish_run_to_hub", _PUBLISH_PATH)
_publish = importlib.util.module_from_spec(_spec)
sys.modules["publish_run_to_hub"] = _publish
_spec.loader.exec_module(_publish)

METRICS_BEGIN = _publish.METRICS_BEGIN
METRICS_END = _publish.METRICS_END
build_metrics_markdown = _publish.build_metrics_markdown
patch_readme_metrics = _publish.patch_readme_metrics


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
            str(_PUBLISH_PATH),
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
            str(_PUBLISH_PATH),
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


def test_default_release_tag_layouts(tmp_path: Path):
    tag = _publish.default_release_tag
    assert tag(tmp_path / "runs" / "m9a-x" / "latest") == "m9a-x"
    # Modal per-run archive /checkpoints/runs/<id>: the id, never "runs"
    assert tag(tmp_path / "checkpoints" / "runs" / "rid-7") == "rid-7"
    # layouts that name no run fall back to summary run_id (caller)
    assert tag(tmp_path / "checkpoints" / "latest") == ""
    assert tag(tmp_path / "artifacts" / "pytorch" / "model") == ""


def test_artifact_mismatch_detects_other_weights(tmp_path: Path):
    import hashlib

    (tmp_path / "model.safetensors").write_bytes(b"weights-epoch-3")
    good = hashlib.sha256(b"weights-epoch-3").hexdigest()
    assert _publish.artifact_mismatch(
        {"artifact_sha": good, "model_kind": "pytorch"}, tmp_path) is None
    assert "artifact_sha" in _publish.artifact_mismatch(
        {"artifact_sha": "0" * 64, "model_kind": "pytorch"}, tmp_path)
    # ONNX evals hash the graph, not the safetensors: not comparable
    assert _publish.artifact_mismatch(
        {"artifact_sha": "0" * 64, "model_kind": "onnx-int8"}, tmp_path) is None


def test_readme_only_404_counts_as_missing():
    class _Resp:
        def __init__(self, code):
            self.status_code = code

    class _HTTPErr(Exception):
        def __init__(self, code):
            super().__init__(code)
            self.response = _Resp(code)

    assert _publish._is_missing_entry(_HTTPErr(404)) is True
    assert _publish._is_missing_entry(_HTTPErr(500)) is False
    assert _publish._is_missing_entry(TimeoutError()) is False


def test_invoke_check_gates_cannot_override_failing_gates(tmp_path: Path, monkeypatch):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps({"doc_type_accuracy": 0.10}), encoding="utf-8")
    monkeypatch.setattr(_publish.subprocess, "call", lambda *a, **k: 0)
    rc = _publish.main(["--dry-run", "--invoke-check-gates", "--checkpoint",
                        str(ckpt), "--eval-json", str(eval_path),
                        "--release-tag", "t"])
    assert rc == 1


def test_cli_refuses_eval_of_other_weights(tmp_path: Path, capsys):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"w")
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps({
        "doc_type_accuracy": 0.9, "window_calibration": {"ece": 0.01},
        "per_head": {"contract": {"macro_f1": 0.3},
                     "correspondence": {"macro_f1": 0.3}},
        "artifact_sha": "0" * 64, "model_kind": "pytorch",
    }), encoding="utf-8")
    argv = ["--dry-run", "--checkpoint", str(ckpt), "--eval-json",
            str(eval_path), "--release-tag", "t"]
    assert _publish.main(argv) == 1
    assert "does not describe these weights" in capsys.readouterr().err
    assert _publish.main(argv + ["--allow-artifact-mismatch"]) == 0
