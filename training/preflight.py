#!/usr/bin/env python3
"""Operator preflight checks before Hub publish or Modal GPU spend.

    uv run python training/preflight.py
    uv run python training/preflight.py --online   # needs HF_TOKEN
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_b = Path(__file__).resolve()
while not (_b / ".git").is_dir():
    _parent = _b.parent
    if _parent == _b:
        raise SystemExit(f"repo root not found above {Path(__file__).resolve()}")
    _b = _parent
ROOT = _b
sys.path.insert(0, str(ROOT / "src"))

from mailroom_ml import config as cfg  # noqa: E402
from mailroom_ml.dataset import verify_stage  # noqa: E402
from mailroom_ml.synthetic_policy import (  # noqa: E402
    assert_policy_matches_config,
    load_synthetic_policy,
    policy_version,
)


def _check_stage(stage: Path) -> list[str]:
    errors: list[str] = []
    if not stage.is_dir():
        errors.append(f"stage missing: {stage} (run build_dataset --stage-only)")
        return errors
    try:
        verify_stage(stage)
    except Exception as exc:  # noqa: BLE001 — operator CLI surfaces all failures
        errors.append(f"verify_stage failed: {exc}")
    return errors


def _check_hub_pin(online: bool) -> list[str]:
    if not online:
        return []
    import os

    if not os.environ.get("HF_TOKEN"):
        return ["--online set but HF_TOKEN is absent"]
    from huggingface_hub import HfApi  # noqa: PLC0415

    errors: list[str] = []
    api = HfApi()
    for label, repo, rev in (
        ("training_data", cfg.TRAINING_DATA_REPO, cfg.TRAINING_DATA_REVISION),
        ("finetune_corpus", cfg.FINETUNE_REPO, cfg.FINETUNE_REVISION),
    ):
        info = api.dataset_info(repo, revision=rev)
        sha = str(getattr(info, "sha", ""))
        if not sha.startswith(rev):
            errors.append(
                f"{label} pin mismatch: config {rev[:12]}… vs Hub {sha[:12]}…"
            )
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage",
        type=Path,
        default=cfg.STAGE_DIR,
        help="staged training tree to verify",
    )
    ap.add_argument(
        "--online",
        action="store_true",
        help="verify Hub dataset pins resolve (HF_TOKEN required)",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=cfg.REPORTS_DIR / "preflight_latest.json",
        help="write JSON report (default: reports/preflight_latest.json)",
    )
    args = ap.parse_args()

    report: dict[str, object] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "checks": {},
        "errors": [],
    }

    try:
        policy = load_synthetic_policy()
        assert_policy_matches_config(policy)
        report["checks"]["synthetic_policy"] = {
            "ok": True,
            "version": policy_version(policy),
            "path": str(cfg.ROOT / "configs" / "synthetic_policy_v1.yaml"),
        }
    except Exception as exc:  # noqa: BLE001
        report["checks"]["synthetic_policy"] = {"ok": False, "error": str(exc)}
        report["errors"].append(str(exc))

    stage_errors = _check_stage(args.stage)
    report["checks"]["stage"] = {"ok": not stage_errors, "path": str(args.stage)}
    report["errors"].extend(stage_errors)

    hub_errors = _check_hub_pin(args.online)
    report["checks"]["hub_pins"] = {"ok": not hub_errors, "online": args.online}
    report["errors"].extend(hub_errors)

    report["ok"] = not report["errors"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if report["errors"]:
        for err in report["errors"]:
            print(f"FAIL: {err}", file=sys.stderr)
        print(f"report: {args.report}", file=sys.stderr)
        return 1
    print(f"preflight OK — report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
