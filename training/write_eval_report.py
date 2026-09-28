#!/usr/bin/env python3
"""Write TEST-EVAL / M9a markdown from eval JSON (artifact-driven).

Usage:
  uv run python training/write_eval_report.py test --run-tag TAG --eval-json PATH
  uv run python training/write_eval_report.py heldout-plus --run-tag TAG --eval-json PATH
  uv run python training/write_eval_report.py compare --a A.json --b B.json --out PATH.md
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from mailroom_ml.eval_report_md import (  # noqa: E402
    ReportContext,
    render_compare_report,
    render_heldout_plus_report,
    render_test_eval_report,
    render_training_report,
)
from mailroom_ml.m9a_gates import gate_status, gates_all_met, m9a_gate_rows  # noqa: E402

TEST_EVAL = ROOT / "reports" / "TEST-EVAL"
MANIFEST = TEST_EVAL / "MANIFEST.json"


def _load_json(path: Path | None) -> dict:
    if path and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _gates_snapshot(report: dict) -> dict:
    snap = {}
    for name, actual, thr, how in m9a_gate_rows(report):
        key = name.replace(" test macro-F1", "_macro_f1").replace(" ", "_").lower()
        key = key.replace("doc_type_test_accuracy", "doc_type_accuracy")
        key = key.replace("window_ece_(doc_type_calibrated)", "window_ece_calibrated")
        snap[key] = {
            "actual": actual,
            "threshold": thr,
            "how": how,
            "met": gate_status(actual, thr, how) == "MET",
        }
    return snap


def _write_gates_file(eval_path: Path, summary_path: Path | None, arm: str) -> Path | None:
    if not arm:
        return None
    suffix = f"arm{arm.upper()}"
    out = TEST_EVAL / f"gates-check-{suffix}.txt"
    cmd = [sys.executable, str(ROOT / "training" / "check_m9a_gates.py"), str(eval_path)]
    if summary_path and summary_path.is_file():
        cmd.append(str(summary_path))
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)
    out.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return out


def _update_manifest_test(
    *,
    run_tag: str,
    report: dict,
    eval_name: str,
    arm: str,
    role: str,
) -> None:
    if not MANIFEST.is_file():
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = {
        "run_tag": run_tag,
        "role": role,
        "arm": arm,
        "eval_json": f"reports/TEST-EVAL/{eval_name}",
        "eval_json_canonical": f"reports/{eval_name}",
        "report": f"reports/TEST-EVAL/TEST-EVAL-REPORT-{run_tag}.md",
        "training_report": f"reports/M9a-REPORT-{run_tag}.md",
        "artifact_sha": report.get("artifact_sha"),
        "gates_112": _gates_snapshot(report),
        "gates_all_met": gates_all_met(report),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    runs = manifest.get("runs") or []
    runs = [r for r in runs if not (r.get("run_tag") == run_tag and r.get("role") == role)]
    runs.append(entry)
    manifest["runs"] = runs
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _update_manifest_heldout_plus(run_tag: str, eval_name: str, report: dict) -> None:
    if not MANIFEST.is_file():
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = {
        "run_tag": run_tag,
        "role": "heldout-plus-extended",
        "eval_json": f"reports/TEST-EVAL/{eval_name}",
        "eval_json_canonical": f"reports/{eval_name}",
        "report": f"reports/TEST-EVAL/TEST-EVAL-REPORT-heldout-plus-{run_tag}.md",
        "compare_323": f"reports/TEST-EVAL/TEST-EVAL-COMPARE-323-vs-1323-{run_tag}.md",
        "n_docs": report.get("n_docs"),
        "eval_subset": report.get("eval_subset"),
        "artifact_sha": report.get("artifact_sha"),
        "checkpoint": report.get("checkpoint"),
        "note": "Extended pool; not used for #112 gate replacement.",
        "comparable_metrics_schema": "mailroom-ml/comparable-metrics/v1",
    }
    runs = manifest.get("runs") or []
    runs = [r for r in runs if not (r.get("run_tag") == run_tag and r.get("role") == entry["role"])]
    runs.append(entry)
    manifest["runs"] = runs
    manifest["heldout_plus_eval"] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _baseline_from_manifest() -> dict[str, float]:
    if not MANIFEST.is_file():
        return {}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    baselines = manifest.get("baselines") or {}
    out: dict[str, float] = {}
    if baselines.get("run3_doc_type_accuracy") is not None:
        out["run-3 (`eval_run3_20260921.json`)"] = float(baselines["run3_doc_type_accuracy"])
    for r in manifest.get("runs") or []:
        if r.get("arm") == "A" and r.get("gates_112"):
            # prefer stored eval if present
            pass
    return out


def cmd_test(args: argparse.Namespace) -> int:
    eval_path = args.eval_json.resolve()
    report = _load_json(eval_path)
    summary = _load_json(args.summary_json)
    run_tag = args.run_tag
    TEST_EVAL.mkdir(parents=True, exist_ok=True)
    eval_name = eval_path.name
    te_copy = TEST_EVAL / eval_name
    if eval_path != te_copy:
        shutil.copy2(eval_path, te_copy)

    gates_path = _write_gates_file(eval_path, args.summary_json, args.arm)
    gates_rel = ""
    if gates_path:
        gates_rel = str(gates_path.relative_to(ROOT))

    baselines = _baseline_from_manifest()
    if report.get("doc_type_accuracy") is not None:
        label = f"**this run ({args.arm or 'primary'})**"
        baselines[label] = float(report["doc_type_accuracy"])
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for r in manifest.get("runs") or []:
            if r.get("arm") == "A" and r.get("eval_json"):
                ej = ROOT / str(r["eval_json"]).replace("reports/TEST-EVAL/", "reports/TEST-EVAL/")
                canon = ROOT / str(r.get("eval_json_canonical", ""))
                path = canon if canon.is_file() else (ROOT / r["eval_json"])
                if path.is_file():
                    br = _load_json(path)
                    if br.get("doc_type_accuracy") is not None:
                        baselines["Arm A smoke"] = float(br["doc_type_accuracy"])
                    break

    ctx = ReportContext(
        run_tag=run_tag,
        arm=args.arm or "",
        run_id=args.run_id or summary.get("run_id", ""),
        checkpoint=str(report.get("checkpoint") or f"data/modernbert_training/runs/{run_tag}/latest"),
        training_log=args.training_log or f"logs/{run_tag}.log",
        gates_check_file=gates_rel,
        baseline_doc_type=baselines,
    )
    md = render_test_eval_report(
        report,
        ctx=ctx,
        summary=summary or None,
        eval_json_canonical=f"reports/{eval_name}",
        eval_json_test_eval=f"reports/TEST-EVAL/{eval_name}",
    )
    out_md = TEST_EVAL / f"TEST-EVAL-REPORT-{run_tag}.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"wrote {out_md}", flush=True)

    if (summary and summary.keys()) or args.write_training_report:
        train_md = render_training_report(report, ctx=ctx, summary=summary or None)
        train_path = ROOT / "reports" / f"M9a-REPORT-{run_tag}.md"
        train_path.write_text(train_md, encoding="utf-8")
        print(f"wrote {train_path}", flush=True)

    _update_manifest_test(
        run_tag=run_tag,
        report=report,
        eval_name=eval_name,
        arm=args.arm or "",
        role=args.role or "m9a-held-out-test",
    )
    return 0


def cmd_heldout_plus(args: argparse.Namespace) -> int:
    eval_path = args.eval_json.resolve()
    report = _load_json(eval_path)
    run_tag = args.run_tag
    plus_fn: set[str] = set()
    if args.plus_dir and (args.plus_dir / "documents.parquet").is_file():
        import pandas as pd  # noqa: PLC0415

        plus_fn = set(pd.read_parquet(args.plus_dir / "documents.parquet")["filename"].astype(str))

    TEST_EVAL.mkdir(parents=True, exist_ok=True)
    eval_name = eval_path.name
    te_copy = TEST_EVAL / eval_name
    if eval_path != te_copy:
        shutil.copy2(eval_path, te_copy)

    ctx = ReportContext(
        run_tag=run_tag,
        checkpoint=str(report.get("checkpoint", "")),
    )
    md = render_heldout_plus_report(
        report,
        ctx=ctx,
        plus_filenames=plus_fn,
        eval_json_name=eval_name,
    )
    out_md = TEST_EVAL / f"TEST-EVAL-REPORT-heldout-plus-{run_tag}.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"wrote {out_md}", flush=True)

    baseline = _load_json(args.baseline_json) if args.baseline_json else {}
    if baseline:
        from compare_runs import compare_reports, load_eval_json  # noqa: E402

        cmp_payload = compare_reports(baseline, report, cohort="")
        body = render_compare_report(
            cmp_payload,
            label_a="canonical323",
            label_b="heldout1323",
            path_a=str(args.baseline_json),
            path_b=str(eval_path),
            title="",
        )
        compare_path = TEST_EVAL / f"TEST-EVAL-COMPARE-323-vs-1323-{run_tag}.md"
        compare_path.write_text(
            "\n".join([
                f"# TEST-EVAL — 323 vs 1,323 (`{run_tag}`)",
                "",
                "Paired bootstrap uses **323 overlapping filenames** only "
                "(canonical test block).",
                "",
                body,
            ]),
            encoding="utf-8",
        )
        print(f"wrote {compare_path}", flush=True)

    _update_manifest_heldout_plus(run_tag, eval_name, report)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from compare_runs import compare_reports, load_eval_json  # noqa: E402

    report_a = load_eval_json(args.a)
    report_b = load_eval_json(args.b)
    cmp = compare_reports(
        report_a, report_b,
        n_resamples=args.resamples,
        seed=args.seed,
        cohort=args.cohort or "",
    )
    body = render_compare_report(
        cmp,
        label_a=args.label_a,
        label_b=args.label_b,
        path_a=str(args.a),
        path_b=str(args.b),
        title="",
    )
    md = f"{args.title}\n\n{body}" if args.title else body
    out = args.out or Path(f"reports/TEST-EVAL/compare-{args.label_a}-vs-{args.label_b}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("test", help="323-doc TEST-EVAL report")
    t.add_argument("--run-tag", required=True)
    t.add_argument("--eval-json", type=Path, required=True)
    t.add_argument("--summary-json", type=Path, default=None)
    t.add_argument("--arm", default="")
    t.add_argument("--run-id", default="")
    t.add_argument("--training-log", default="")
    t.add_argument("--baseline-json", type=Path, default=None)
    t.add_argument("--role", default="m9a-held-out-test")
    t.add_argument("--write-training-report", action="store_true")
    t.set_defaults(func=cmd_test)

    h = sub.add_parser("heldout-plus", help="1,323-doc heldout-plus report")
    h.add_argument("--run-tag", required=True)
    h.add_argument("--eval-json", type=Path, required=True)
    h.add_argument("--baseline-json", type=Path, default=None)
    h.add_argument("--plus-dir", type=Path, default=Path("data/heldout_plus_v1"))
    h.set_defaults(func=cmd_heldout_plus)

    c = sub.add_parser("compare", help="compare_runs markdown")
    c.add_argument("--a", type=Path, required=True)
    c.add_argument("--b", type=Path, required=True)
    c.add_argument("--out", type=Path, default=None)
    c.add_argument("--label-a", default="A")
    c.add_argument("--label-b", default="B")
    c.add_argument("--title", default="")
    c.add_argument("--cohort", default="")
    c.add_argument("--resamples", type=int, default=2000)
    c.add_argument("--seed", type=int, default=42)
    c.set_defaults(func=cmd_compare)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
