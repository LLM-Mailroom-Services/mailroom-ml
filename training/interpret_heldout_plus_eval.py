#!/usr/bin/env python3
"""Write TEST-EVAL markdown + MANIFEST entry from a heldout-plus eval JSON."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-json", type=Path, required=True)
    ap.add_argument("--baseline-json", type=Path, default=None)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--plus-dir", type=Path, default=Path("data/heldout_plus_v1"))
    args = ap.parse_args(argv)

    cmd = [
        sys.executable,
        str(ROOT / "training" / "write_eval_report.py"),
        "heldout-plus",
        "--run-tag",
        args.run_tag,
        "--eval-json",
        str(args.eval_json),
        "--plus-dir",
        str(args.plus_dir),
    ]
    if args.baseline_json and args.baseline_json.is_file():
        cmd.extend(["--baseline-json", str(args.baseline_json)])
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
