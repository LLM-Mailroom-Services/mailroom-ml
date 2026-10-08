#!/usr/bin/env python3
"""Publish a trained ModernBERT checkpoint + held-out test eval to the Hub.

Operator-only: requires ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``). Use
``--dry-run`` to print the upload plan without network I/O.

Typical post-M9a sequence (after ``run_m9a_local.sh`` train + eval + gates):

    HF_TOKEN=... .venv/bin/python training/train/post-train/publish_run_to_hub.py \\
      --checkpoint data/modernbert_training/runs/m9a-local-.../latest \\
      --eval-json reports/eval_m9a-local-....json \\
      --release-tag m9a-local-20260927-010430

See ``governance/M9a-HANDOFF.md`` § post-train Hub release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from mailroom_ml.config import CLASSIFIER_MODEL_REPO
from mailroom_ml.m9a_gates import gate_status, gates_all_met, m9a_gate_rows

IGNORE_UPLOAD = ["optimizer.pt", "scheduler.pt", "resume.json"]
METRICS_BEGIN = "<!-- mailroom-ml:test-metrics:begin -->"
METRICS_END = "<!-- mailroom-ml:test-metrics:end -->"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


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
            f"{gate_status(actual, thr, how)} |"
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


def default_release_tag(checkpoint: Path) -> str:
    """Release tag implied by the checkpoint layout ("" when none is).

    ``.../runs/<tag>/latest`` (local runs) -> ``<tag>``;
    ``.../runs/<tag>`` (the Modal per-run archive) -> ``<tag>``.  Any other
    layout (``artifacts/pytorch/model``, ``/checkpoints/latest``) names no
    run, so the caller falls back to ``summary.json`` ``run_id`` instead of
    tagging the release ``pytorch`` / ``checkpoints`` / ``runs``.
    """
    ck = checkpoint.resolve()
    if ck.name == "latest" and ck.parent.parent.name == "runs":
        return ck.parent.name
    if ck.parent.name == "runs":
        return ck.name
    return ""


def _sha256_file(path: Path) -> str:
    """Return the file contents' SHA-256 hex digest; file I/O errors propagate."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_mismatch(report: dict, checkpoint: Path) -> str | None:
    """Why the eval JSON does not describe these weights, or None.

    ``eval_modernbert`` records ``artifact_sha`` = sha256 of the evaluated
    ``model.safetensors`` (pytorch bundles).  Publishing an eval of another
    epoch / run next to these weights would ship a model card that lies.

    Returns None when the hash is absent, the eval is not PyTorch, or the
    checkpoint has no ``model.safetensors`` file; these cases are unchecked.
    Errors reading an existing weights file propagate.
    """
    want = report.get("artifact_sha")
    weights = checkpoint / "model.safetensors"
    if not want or report.get("model_kind") != "pytorch" or not weights.is_file():
        return None
    have = _sha256_file(weights)
    if have != want:
        return (f"eval artifact_sha {str(want)[:12]}… != checkpoint "
                f"model.safetensors {have[:12]}…")
    return None


def _is_missing_entry(exc: Exception) -> bool:
    """Return whether the error is a Hub missing-entry error or has HTTP status 404."""
    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:  # older huggingface_hub
        from huggingface_hub.utils import EntryNotFoundError
    if isinstance(exc, EntryNotFoundError):
        return True
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None) == 404


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for checkpoint publication and upload-plan validation."""
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
    ap.add_argument(
        "--allow-artifact-mismatch",
        action="store_true",
        help="upload even when the eval JSON's artifact_sha does not match "
             "the checkpoint's model.safetensors",
    )
    return ap


def _token_present() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    """Validate a release and optionally upload its checkpoint, eval, and model card.

    ``argv=None`` reads process arguments. Returns 2 for missing input paths,
    1 for a refused gate check or artifact mismatch, and 0 for successful
    uploads or a valid dry run. Missing credentials also select a dry run;
    gate and artifact checks still apply.

    File-read, JSON-decoding, and Hub upload errors propagate. A missing
    README starts a new model card; other download errors propagate.
    Tag creation failures are caught after uploads and still return 0.
    """
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

    release_tag = (args.release_tag or "").strip() or default_release_tag(ckpt)
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
    mismatch = artifact_mismatch(report, ckpt)

    def _commit_msg() -> str:
        """Describe the release using the current gate result for the upload commit."""
        return (
            f"Release {release_tag} ({released_at}): ModernBERT classifier "
            f"run_id={run_id} M9a gates={'PASS' if gates_ok else 'FAIL'}"
        )

    commit_msg = _commit_msg()

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
        # training/train/post-train/ -> training/check_m9a_gates.py
        gate_script = Path(__file__).resolve().parents[2] / "check_m9a_gates.py"
        rc = subprocess.call(
            [sys.executable, str(gate_script), str(eval_path), str(summary_path)]
            if summary_path.is_file()
            else [sys.executable, str(gate_script), str(eval_path)],
        )
        if rc != 0 and not args.ignore_gates:
            print("ERROR: check_m9a_gates.py failed — refusing upload", file=sys.stderr)
            return 1
        # the script can only veto: a passing script never overrides a failing
        # in-process gate check (both read the same eval JSON)
        gates_ok = gates_ok and rc == 0
        commit_msg = _commit_msg()

    if mismatch and not args.allow_artifact_mismatch:
        print(f"ERROR: {mismatch} — the eval does not describe these weights; "
              "pass --allow-artifact-mismatch to publish anyway",
              file=sys.stderr)
        return 1

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
    except Exception as exc:
        # only "no README yet" starts an empty card; a transient / auth error
        # must not overwrite the existing model card with just the metrics
        if not _is_missing_entry(exc):
            raise
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
