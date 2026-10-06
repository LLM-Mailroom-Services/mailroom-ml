"""Shell sanity for training/complete_run.sh (no GPU / no train side effects)."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "training" / "train" / "complete_run.sh"


def test_complete_run_bash_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True, cwd=ROOT)


def test_complete_run_help():
    proc = subprocess.run(
        [str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--run-tag" in proc.stdout
    assert "complete_run" in proc.stdout or "Post-train" in proc.stdout


def test_complete_run_missing_artifacts_read_only():
    proc = subprocess.run(
        [str(SCRIPT), "--run-tag", "m9a-local-nonexistent-tag-000000"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 4
    assert "summary.json" in proc.stderr or "summary.json" in proc.stdout
    assert "WAIT" in proc.stderr or "WAIT" in proc.stdout
