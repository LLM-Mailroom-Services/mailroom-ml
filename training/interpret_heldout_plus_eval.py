#!/usr/bin/env python3
"""Write TEST-EVAL markdown + MANIFEST entry from a heldout-plus eval JSON."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from compare_runs import compare_reports, format_markdown, load_eval_json  # noqa: E402
from eval_modernbert import build_comparable_metrics  # noqa: E402

TEST_EVAL = ROOT / "reports" / "TEST-EVAL"
MANIFEST = TEST_EVAL / "MANIFEST.json"


def _slice_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n_docs": 0}
    dt_ok = sum(1 for r in rows if r.get("dt_correct"))
    sc_rows = [r for r in rows if r.get("dt_correct")]
    sc_ok = sum(1 for r in sc_rows if r.get("sc_correct"))
    by_dt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_dt[str(r.get("gt_doc_type"))].append(r)
    per_type = {}
    for dt, rs in sorted(by_dt.items()):
        d_ok = sum(1 for r in rs if r.get("dt_correct"))
        per_type[dt] = {
            "n": len(rs),
            "doc_type_accuracy": round(d_ok / len(rs), 4) if rs else None,
        }
    return {
        "n_docs": n,
        "doc_type_accuracy": round(dt_ok / n, 4),
        "subclass_accuracy_conditional": round(sc_ok / len(sc_rows), 4)
        if sc_rows else None,
        "per_doc_type": per_type,
    }


def _macro_f1_correspondence(rows: list[dict]) -> float | None:
    pairs = []
    for r in rows:
        if str(r.get("gt_doc_type")) != "correspondence":
            continue
        if not r.get("dt_correct"):
            continue
        pairs.append((str(r.get("gt_subclass")), r.get("pred_subclass")))
    if not pairs:
        return None
    classes = sorted({gt for gt, _ in pairs})
    f1s = []
    for c in classes:
        tp = sum(1 for gt, pred in pairs if gt == c and pred == c)
        fp = sum(1 for gt, pred in pairs if gt != c and pred == c)
        fn = sum(1 for gt, pred in pairs if gt == c and pred != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return round(float(sum(f1s) / len(f1s)), 4)


def _write_report(
    *,
    run_tag: str,
    report: dict,
    baseline: dict | None,
    plus_filenames: set[str],
    out_md: Path,
    compare_md: Path,
) -> None:
    per_doc = report.get("per_doc") or []
    canon_rows = [r for r in per_doc if str(r.get("filename")) not in plus_filenames]
    plus_rows = [r for r in per_doc if str(r.get("filename")) in plus_filenames]
    canon_m = _slice_metrics(canon_rows)
    plus_m = _slice_metrics(plus_rows)
    corr_canon = _macro_f1_correspondence(canon_rows)
    corr_plus = _macro_f1_correspondence(plus_rows)

    wc = report.get("window_calibration") or {}
    lines = [
        f"# TEST-EVAL — held-out-plus (`{run_tag}`)",
        "",
        "**Eval role:** extended monitoring pool (1,323 docs); **#112 gates remain "
        "on canonical 323 only** (`--subset test`).",
        f"**Run tag:** `{run_tag}`",
        f"**Eval JSON:** `{out_md.stem.replace('TEST-EVAL-REPORT-heldout-plus-', 'eval_')}.json`",
        f"**Checkpoint:** `{report.get('checkpoint', '')}`",
        f"**Artifact SHA:** `{report.get('artifact_sha', '')}`",
        "",
        "## Harness protocol",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        "| CLI | `training/eval_modernbert.py` |",
        "| `--subset` | `heldout-plus` |",
        f"| Documents | **{report.get('n_docs')}** (`--sample 0`) |",
        f"| Windows | **{wc.get('n_windows')}** |",
        "| `--max-length` | 8192 |",
        "| seed | 42 |",
        "",
        "## Headline metrics (full 1,323)",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| doc_type_accuracy | **{report.get('doc_type_accuracy')}** |",
        f"| subclass_accuracy_conditional | **{report.get('subclass_accuracy_conditional')}** |",
        f"| window ECE | {wc.get('ece')} |",
        f"| fast_path_rate | {report.get('fast_path_rate')} |",
        "",
        "## Slices (canonical 323 vs Enron plus 1,000)",
        "",
        "| slice | n | doc_type_acc | subclass (cond.) | corr. macro-F1 (cond.) |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| canonical test | {canon_m.get('n_docs')} | {canon_m.get('doc_type_accuracy')} "
        f"| {canon_m.get('subclass_accuracy_conditional')} | {corr_canon} |",
        f"| plus v1 (Enron) | {plus_m.get('n_docs')} | {plus_m.get('doc_type_accuracy')} "
        f"| {plus_m.get('subclass_accuracy_conditional')} | {corr_plus} |",
        "",
        "### Per doc_type (plus slice)",
        "",
        "| doc_type | n | doc_type_acc |",
        "| --- | ---: | ---: |",
    ]
    for dt, info in (plus_m.get("per_doc_type") or {}).items():
        lines.append(f"| {dt} | {info['n']} | {info['doc_type_accuracy']} |")
    lines += [
        "",
        "## Per-head macro-F1 (full pool, harness)",
        "",
    ]
    for head, m in (report.get("per_head") or {}).items():
        lines.append(f"- **{head}**: macro_f1={m.get('macro_f1')}")
    cm = report.get("comparable_metrics") or build_comparable_metrics(report)
    lines += [
        "",
        "## LLM sorter comparable metrics (`compare_runs` / shadow eval)",
        "",
        "Structured block also embedded in eval JSON as `comparable_metrics` "
        "(schema `mailroom-ml/comparable-metrics/v1`). Pair with an LLM sorter "
        "eval export via `training/compare_runs.py` on shared filenames.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| doc_type_accuracy | {cm.get('doc_type_accuracy')} |",
        f"| subclass_accuracy_conditional | {cm.get('subclass_accuracy_conditional')} |",
        f"| window_ece | {cm.get('window_ece')} |",
        f"| window_band_ece | {cm.get('window_band_ece')} |",
        f"| mean_window_agreement | {cm.get('mean_window_agreement')} |",
        f"| fast_path_rate | {cm.get('fast_path_rate')} |",
        f"| fast_path_n_docs | {cm.get('fast_path_n_docs')} |",
        f"| ood_rate | {cm.get('ood_rate')} |",
    ]
    sr = cm.get("selective_risk") or {}
    if sr.get("refused"):
        lines.append(f"| selective_risk | REFUSED: {sr.get('reason')} |")
    elif sr:
        lines.append(
            f"| selective_risk threshold | {sr.get('recommended_threshold')} "
            f"(budget_met={sr.get('budget_met')}, coverage={sr.get('coverage')}) |"
        )
    lat = cm.get("latency_seconds_per_document")
    if lat is not None:
        lines.append(f"| latency_s_per_doc (Modal) | {lat} |")
    lines += [
        "",
        "### Per-head macro-F1 (comparable block)",
        "",
    ]
    for head, mf1 in (cm.get("per_head_macro_f1") or {}).items():
        lines.append(f"- **{head}**: {mf1}")
    if baseline:
        cmp_payload = compare_reports(baseline, report, cohort="")
        cmp_text = format_markdown(
            cmp_payload, label_a="canonical323", label_b="heldout1323")
        compare_md.write_text(
            "\n".join([
                f"# TEST-EVAL — 323 vs 1,323 (`{run_tag}`)",
                "",
                "Paired bootstrap uses **323 overlapping filenames** only "
                "(canonical test block).",
                "",
                cmp_text,
                "",
            ]),
            encoding="utf-8",
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_manifest(run_tag: str, eval_name: str, report: dict) -> None:
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
        "note": "Extended pool; not used for #112 gate replacement.",
        "comparable_metrics_schema": "mailroom-ml/comparable-metrics/v1",
        "telemetry_dir": f"reports/TEST-EVAL/telemetry/{eval_name.replace('.json', '')}",
    }
    runs = manifest.get("runs") or []
    runs = [r for r in runs if r.get("run_tag") != run_tag or r.get("role") != entry["role"]]
    runs.append(entry)
    manifest["runs"] = runs
    manifest["heldout_plus_eval"] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-json", type=Path, required=True)
    ap.add_argument("--baseline-json", type=Path, default=None)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--plus-dir", type=Path, default=Path("data/heldout_plus_v1"))
    args = ap.parse_args(argv)

    report = load_eval_json(args.eval_json)
    plus_fn = set()
    pq = args.plus_dir / "documents.parquet"
    if pq.is_file():
        import pandas as pd  # noqa: PLC0415

        plus_fn = set(pd.read_parquet(pq)["filename"].astype(str))
    baseline = load_eval_json(args.baseline_json) if args.baseline_json and args.baseline_json.is_file() else None

    eval_name = args.eval_json.name
    TEST_EVAL.mkdir(parents=True, exist_ok=True)
    te_copy = TEST_EVAL / eval_name
    shutil.copy2(args.eval_json, te_copy)

    report_md = TEST_EVAL / f"TEST-EVAL-REPORT-heldout-plus-{args.run_tag}.md"
    compare_md = TEST_EVAL / f"TEST-EVAL-COMPARE-323-vs-1323-{args.run_tag}.md"
    _write_report(
        run_tag=args.run_tag,
        report=report,
        baseline=baseline,
        plus_filenames=plus_fn,
        out_md=report_md,
        compare_md=compare_md,
    )
    _update_manifest(args.run_tag, eval_name, report)

    holdout = ROOT / "reports" / "HOLDOUT-PLUS-V1-REPORT.md"
    if holdout.is_file():
        stamp = datetime.now(UTC).strftime("%Y-%m-%d")
        note = (
            f"\n## GPU validation ({stamp})\n\n"
            f"- Eval JSON: `{args.eval_json}`\n"
            f"- Report: `{report_md.relative_to(ROOT)}`\n"
            f"- n_docs={report.get('n_docs')} windows="
            f"{(report.get('window_calibration') or {}).get('n_windows')}\n"
        )
        text = holdout.read_text(encoding="utf-8")
        if "## GPU validation" not in text:
            holdout.write_text(text.rstrip() + "\n" + note, encoding="utf-8")

    print(f"wrote {report_md}", flush=True)
    print(f"wrote {te_copy}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
