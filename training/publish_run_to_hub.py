#!/usr/bin/env python3
"""Publish a trained ModernBERT checkpoint + held-out test eval to the Hub.

Operator-only: requires ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``). Use
``--dry-run`` to print the upload plan without network I/O.

Typical post-M9a sequence (after ``run_m9a_local.sh`` train + eval + gates):

    HF_TOKEN=... .venv/bin/python training/publish_run_to_hub.py \\
      --checkpoint data/modernbert_training/runs/m9a-local-.../latest \\
      --eval-json reports/eval_m9a-local-....json \\
      --release-tag m9a-local-20260927-010430

See ``governance/M9a-HANDOFF.md`` § post-train Hub release.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from mailroom_ml.config import CLASSIFIER_MODEL_REPO

IGNORE_UPLOAD = ["optimizer.pt", "scheduler.pt", "resume.json"]
METRICS_BEGIN = "<!-- mailroom-ml:test-metrics:begin -->"
METRICS_END = "<!-- mailroom-ml:test-metrics:end -->"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def m9a_gate_rows(report: dict) -> list[tuple[str, float | None, float, str]]:
    """Same thresholds as ``training/check_m9a_gates.py``."""
    per_head = report.get("per_head") or {}
    contract = _f((per_head.get("contract") or {}).get("macro_f1"))
    corr = _f((per_head.get("correspondence") or {}).get("macro_f1"))
    dt_acc = _f(report.get("doc_type_accuracy"))
    ece = _f((report.get("window_calibration") or {}).get("ece"))
    return [
        ("contract test macro-F1", contract, 0.20, "ge"),
        ("correspondence test macro-F1", corr, 0.25, "ge"),
        ("doc_type test accuracy", dt_acc, 0.89, "ge"),
        ("window ECE (doc_type calibrated)", ece, 0.05, "le"),
    ]


def gates_all_met(report: dict) -> bool:
    for _name, actual, thr, how in m9a_gate_rows(report):
        if actual is None:
            return False
        if how == "ge" and actual < thr:
            return False
        if how == "le" and actual > thr:
            return False
    return True


def _gate_status(actual: float | None, thr: float, how: str) -> str:
    if actual is None:
        return "MISSING"
    if how == "ge":
        return "MET" if actual >= thr else "NOT MET"
    return "MET" if actual <= thr else "NOT MET"


def build_metrics_markdown(
    *,
    release_tag: str,
    released_at: str,
    eval_artifact_name: str,
    report: dict,
    summary: dict,
) -> str:
    lines = [
        f"## Release `{release_tag}` ({released_at})",
        "",
        f"Held-out test eval artifact on this repo: `{eval_artifact_name}`.",
        "",
        "### M9a #112 gates (held-out test)",
        "",
        "| Gate | Actual | Threshold | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, actual, thr, how in m9a_gate_rows(report):
        op = "≥" if how == "ge" else "≤"
        lines.append(
            f"| {name} | {actual} | {op} {thr} | "
            f"{_gate_status(actual, thr, how)} |"
        )
    lines.extend([
        "",
        "### Test surfaces",
        "",
        f"- subclass_accuracy_conditional: "
        f"{report.get('subclass_accuracy_conditional')}",
    ])
    wc = report.get("window_calibration") or {}
    lines.append(
        f"- window_calibration: ece={wc.get('ece')} "
        f"band_ece={wc.get('band_ece')}"
    )
    lines.append("")
    lines.append("### Per-head test macro-F1 (observed)")
    lines.append("")
    lines.append("| head | macro-F1 | support | ECE (calibrated) |")
    lines.append("| --- | ---: | ---: | ---: |")
    per_head = report.get("per_head") or {}
    for head in sorted(per_head):
        info = per_head[head]
        if not isinstance(info, dict):
            continue
        lines.append(
            f"| {head} | {info.get('macro_f1')} | {info.get('support')} | "
            f"{info.get('ece_calibrated')} |"
        )
    sel = summary.get("checkpoint_selection") or {}
    if sel:
        lines.extend([
            "",
            "### Trainer selection (validation)",
            "",
            f"- selected epoch: {sel.get('epoch')}",
            f"- gate_met: {sel.get('gate_met')}",
            f"- subclass_objective: {sel.get('subclass_objective')}",
            f"- doc_type macro-F1 (val): {sel.get('macro_f1')}",
            f"- doc_type ECE (val, calibrated): {sel.get('ece')}",
        ])
        excl = (sel.get("head_exclusion_policy") or {}).get("excluded") or {}
        if excl:
            lines.append("- head_exclusion_policy (ECE > 0.05):")
            for h, meta in sorted(excl.items()):
                if isinstance(meta, dict):
                    lines.append(
                        f"  - {h}: excluded={meta.get('excluded')} "
                        f"ece={meta.get('ece_calibrated')}"
                    )
    lines.append("")
    return "\n".join(lines)


def patch_readme_metrics(existing: str, metrics_block: str) -> str:
    wrapped = (
        f"{METRICS_BEGIN}\n{metrics_block}\n{METRICS_END}"
    )
    if METRICS_BEGIN in existing and METRICS_END in existing:
        pre = existing.split(METRICS_BEGIN, 1)[0]
        post = existing.split(METRICS_END, 1)[1]
        return pre + wrapped + post
    if existing.strip():
        return existing.rstrip() + "\n\n" + wrapped + "\n"
    return (
        "---\n"
        "license: apache-2.0\n"
        "tags:\n"
        "  - text-classification\n"
        "---\n\n"
        "# mailroom ModernBERT hierarchical classifier\n\n"
        + wrapped
        + "\n"
    )


def resolve_run_id(summary: dict, checkpoint: Path, release_tag: str) -> str:
    rid = (summary.get("run_id") or "").strip()
    if rid:
        return rid
    if release_tag:
        return release_tag
    return checkpoint.resolve().name


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="directory with model weights + summary.json")
    ap.add_argument("--eval-json", type=Path, required=True,
                    help="held-out eval report from eval_modernbert.py")
    ap.add_argument("--repo", default=CLASSIFIER_MODEL_REPO,
                    help=f"Hub model repo (default: {CLASSIFIER_MODEL_REPO})")
    ap.add_argument(
        "--release-tag",
        default="",
        help="version tag on the Hub repo (default: run_id or checkpoint parent)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print upload plan; no Hub API calls",
    )
    ap.add_argument(
        "--ignore-gates",
        action="store_true",
        help="upload even when M9a #112 gates are not all MET",
    )
    ap.add_argument(
        "--invoke-check-gates",
        action="store_true",
        help="also run training/check_m9a_gates.py before upload",
    )
    return ap


def _token_present() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ckpt = args.checkpoint.resolve()
    eval_path = args.eval_json.resolve()
    if not ckpt.is_dir():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2
    if not eval_path.is_file():
        print(f"ERROR: eval JSON not found: {eval_path}", file=sys.stderr)
        return 2

    summary_path = ckpt / "summary.json"
    summary = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = json.loads(eval_path.read_text(encoding="utf-8"))

    release_tag = (args.release_tag or "").strip()
    if not release_tag:
        parent = ckpt.parent.name
        if parent and parent != "latest":
            release_tag = parent
    run_id = resolve_run_id(summary, ckpt, release_tag)
    if not release_tag:
        release_tag = run_id

    released_at = datetime.now(UTC).strftime("%Y-%m-%d")
    eval_repo_name = f"eval_report_{release_tag}.json"
    metrics_md = build_metrics_markdown(
        release_tag=release_tag,
        released_at=released_at,
        eval_artifact_name=eval_repo_name,
        report=report,
        summary=summary,
    )

    gates_ok = gates_all_met(report)
    commit_msg = (
        f"Release {release_tag} ({released_at}): ModernBERT classifier "
        f"run_id={run_id} M9a gates={'PASS' if gates_ok else 'FAIL'}"
    )

    plan = {
        "repo": args.repo,
        "repo_url": f"https://huggingface.co/{args.repo}",
        "release_tag": release_tag,
        "run_id": run_id,
        "commit_message": commit_msg,
        "upload_folder": str(ckpt),
        "ignore_patterns": IGNORE_UPLOAD,
        "extra_files": [
            (str(eval_path), eval_repo_name),
            ("README.md (metrics section)", "README.md"),
        ],
        "create_tag": release_tag,
        "m9a_gates": "PASS" if gates_ok else "FAIL",
    }

    print("=== Hub publish plan ===")
    print(json.dumps(plan, indent=2))
    print()
    print(metrics_md)

    if args.invoke_check_gates:
        gate_script = Path(__file__).resolve().parent / "check_m9a_gates.py"
        rc = subprocess.call(
            [sys.executable, str(gate_script), str(eval_path), str(summary_path)]
            if summary_path.is_file()
            else [sys.executable, str(gate_script), str(eval_path)],
        )
        if rc != 0 and not args.ignore_gates:
            print("ERROR: check_m9a_gates.py failed — refusing upload", file=sys.stderr)
            return 1
        gates_ok = rc == 0

    if not gates_ok and not args.ignore_gates:
        print(
            "ERROR: M9a #112 gates not all MET — pass --ignore-gates to "
            "publish anyway",
            file=sys.stderr,
        )
        return 1

    if args.dry_run or not _token_present():
        if not args.dry_run and not _token_present():
            print(
                "DRY RUN (no HF_TOKEN): set HF_TOKEN to execute uploads.",
                file=sys.stderr,
            )
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)

    api.upload_folder(
        folder_path=str(ckpt),
        repo_id=args.repo,
        repo_type="model",
        ignore_patterns=IGNORE_UPLOAD,
        commit_message=commit_msg,
    )

    api.upload_file(
        path_or_fileobj=eval_path.read_bytes(),
        path_in_repo=eval_repo_name,
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"Attach test eval {eval_repo_name} ({release_tag})",
    )

    existing_readme = ""
    try:
        existing_readme = api.hf_hub_download(
            args.repo, "README.md", repo_type="model"
        )
        existing_readme = Path(existing_readme).read_text(encoding="utf-8")
    except Exception:
        existing_readme = ""

    readme_out = patch_readme_metrics(existing_readme, metrics_md)
    api.upload_file(
        path_or_fileobj=readme_out.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"Update model card test metrics ({release_tag})",
    )

    try:
        sha = api.model_info(args.repo).sha
        api.create_tag(
            args.repo,
            tag=release_tag,
            revision=sha,
            repo_type="model",
        )
        print(f"tagged: {args.repo}@{release_tag} -> {sha[:12]}…")
    except Exception as exc:
        print(
            f"WARNING: create_tag failed ({exc}); uploads succeeded",
            file=sys.stderr,
        )

    print(f"pushed: https://huggingface.co/{args.repo}")
    print(f"eval artifact: {eval_repo_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
