#!/usr/bin/env python3
"""Compare two eval JSON reports (classifier vs LLM sorter, or two runs).

Plan §11 / mailroom-ml #17: paired bootstrap CIs on headline metrics.
Does not call live LLM APIs — compare pre-exported eval JSON only.

    uv run python training/compare_runs.py --a reports/eval_a.json \\
        --b reports/eval_b.json
    uv run python training/compare_runs.py --a a.json --b b.json --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # training/<area>/<script>.py -> repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mailroom_ml.config import RANDOM_STATE  # noqa: E402

__all__ = [
    "build_parser",
    "load_eval_json",
    "paired_bootstrap",
    "compare_reports",
    "format_markdown",
    "main",
]

DEFAULT_RESAMPLES = 2000

_SCALAR_KEYS = (
    ("doc_type_accuracy", "doc_type_accuracy"),
    ("window_ece", "window_calibration.ece"),
    ("fast_path_rate", "fast_path_rate"),
    ("selective_risk_threshold", "selective_risk.recommended_threshold"),
    ("ood_rate", "ood.rate"),
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, type=Path,
                    help="eval JSON path A (or a directory containing eval_*.json)")
    ap.add_argument("--b", required=True, type=Path,
                    help="eval JSON path B")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--cohort", default="",
                    help="optional cohort name (single-window|multi-window)")
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES,
                    help="paired-bootstrap resamples (default 2000)")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--json", action="store_true", dest="as_json")
    return ap


def _resolve_eval_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        cands = sorted(path.glob("eval_*.json")) + sorted(path.glob("*.json"))
        if cands:
            return cands[0]
    raise FileNotFoundError(f"no eval JSON at {path}")


def load_eval_json(path: Path) -> dict:
    data = json.loads(_resolve_eval_path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _dig(report: dict, dotted: str):
    cur: object = report
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def paired_bootstrap(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = RANDOM_STATE,
) -> dict:
    """Paired bootstrap on A-B differences (seed pinned, default 2000)."""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if av.shape != bv.shape:
        raise ValueError("paired_bootstrap requires equal-length series")
    if av.size == 0:
        return {
            "mean_diff": None, "ci_low": None, "ci_high": None,
            "n_resamples": n_resamples, "n_paired": 0,
        }
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_resamples, dtype=np.float64)
    n = av.size
    for i in range(n_resamples):
        idx = rng.randint(0, n, n)
        diffs[i] = float(np.mean(av[idx] - bv[idx]))
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_resamples": n_resamples,
        "n_paired": int(n),
    }


def _per_doc_index(report: dict) -> dict[str, dict]:
    rows = report.get("per_doc") or []
    return {str(r.get("filename")): r for r in rows if r.get("filename")}


def _cohort_filter(rows: list[dict], cohort: str) -> list[dict]:
    if not cohort:
        return rows
    if cohort == "single-window":
        return [r for r in rows if int(r.get("n_windows") or 0) == 1]
    if cohort == "multi-window":
        return [r for r in rows if int(r.get("n_windows") or 0) > 1]
    return rows


def compare_reports(
    report_a: dict,
    report_b: dict,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = RANDOM_STATE,
    cohort: str = "",
) -> dict:
    """Headline scalars + paired bootstrap where document IDs align."""
    idx_a = _per_doc_index(report_a)
    idx_b = _per_doc_index(report_b)
    shared = sorted(set(idx_a) & set(idx_b))
    if cohort:
        shared_rows_a = _cohort_filter([idx_a[k] for k in shared], cohort)
        shared = [str(r["filename"]) for r in shared_rows_a
                  if str(r["filename"]) in idx_b]

    metrics: dict[str, dict] = {}
    for name, dotted in _SCALAR_KEYS:
        va = _dig(report_a, dotted)
        vb = _dig(report_b, dotted)
        if cohort and name in {"doc_type_accuracy", "window_ece"}:
            ca = (report_a.get("cohorts") or {}).get(cohort) or {}
            cb = (report_b.get("cohorts") or {}).get(cohort) or {}
            if name == "doc_type_accuracy":
                va, vb = ca.get("doc_type_accuracy"), cb.get("doc_type_accuracy")
            elif name == "window_ece":
                va, vb = ca.get("window_ece"), cb.get("window_ece")
        entry: dict = {
            "a": va, "b": vb,
            "delta": (None if va is None or vb is None
                      else float(va) - float(vb)),
            "paired": None,
        }
        metrics[name] = entry

    per_head: dict[str, dict] = {}
    heads = sorted(set(report_a.get("per_head") or {})
                   | set(report_b.get("per_head") or {}))
    for head in heads:
        va = ((report_a.get("per_head") or {}).get(head) or {}).get("macro_f1")
        vb = ((report_b.get("per_head") or {}).get(head) or {}).get("macro_f1")
        per_head[head] = {
            "a": va, "b": vb,
            "delta": (None if va is None or vb is None
                      else float(va) - float(vb)),
        }

    paired: dict[str, dict] = {}
    if shared:
        dt_a = [1.0 if idx_a[k]["dt_correct"] else 0.0 for k in shared]
        dt_b = [1.0 if idx_b[k]["dt_correct"] else 0.0 for k in shared]
        paired["doc_type_accuracy"] = paired_bootstrap(
            dt_a, dt_b, n_resamples=n_resamples, seed=seed)
        metrics["doc_type_accuracy"]["paired"] = paired["doc_type_accuracy"]
        if all("sc_correct" in idx_a[k] for k in shared):
            sc_a = [1.0 if idx_a[k]["sc_correct"] else 0.0 for k in shared]
            sc_b = [1.0 if idx_b[k]["sc_correct"] else 0.0 for k in shared]
            paired["subclass_accuracy"] = paired_bootstrap(
                sc_a, sc_b, n_resamples=n_resamples, seed=seed)
        if all("fast_path" in idx_a[k] for k in shared):
            fp_a = [1.0 if idx_a[k].get("fast_path") else 0.0 for k in shared]
            fp_b = [1.0 if idx_b[k].get("fast_path") else 0.0 for k in shared]
            paired["fast_path_rate"] = paired_bootstrap(
                fp_a, fp_b, n_resamples=n_resamples, seed=seed)
            metrics["fast_path_rate"]["paired"] = paired["fast_path_rate"]

    return {
        "n_docs_a": report_a.get("n_docs"),
        "n_docs_b": report_b.get("n_docs"),
        "n_paired": len(shared),
        "cohort": cohort or None,
        "seed": seed,
        "resamples": n_resamples,
        "metrics": metrics,
        "per_head": per_head,
        "paired": paired,
        "pairing": "document_id" if shared else "none",
        "note": (
            "Paired bootstrap on aligned per_doc filenames when present; "
            "otherwise headline scalars only (no fabricated pairing)."
        ),
    }


def format_markdown(cmp: dict, *, label_a: str = "A",
                    label_b: str = "B") -> str:
    lines = [
        f"# compare_runs: {label_a} vs {label_b}",
        "",
        f"n_docs {label_a}={cmp['n_docs_a']}  {label_b}={cmp['n_docs_b']}  "
        f"paired={cmp['n_paired']}  pairing={cmp['pairing']}",
        f"bootstrap resamples={cmp['resamples']} seed={cmp['seed']}",
        "",
        f"| metric | {label_a} | {label_b} | delta | 95% CI (A-B) |",
        "|---|---:|---:|---:|---|",
    ]
    for name, row in cmp["metrics"].items():
        ci = row.get("paired") or {}
        ci_s = (
            f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
            if ci.get("ci_low") is not None else "—"
        )
        lines.append(
            f"| {name} | {_fmt(row['a'])} | {_fmt(row['b'])} | "
            f"{_fmt(row['delta'])} | {ci_s} |"
        )
    lines += ["", "per-head macro-F1 (observed):",
              f"| head | {label_a} | {label_b} | delta |",
              "|---|---:|---:|---:|"]
    for head, row in cmp["per_head"].items():
        lines.append(
            f"| {head} | {_fmt(row['a'])} | {_fmt(row['b'])} | "
            f"{_fmt(row['delta'])} |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_a = load_eval_json(args.a)
    report_b = load_eval_json(args.b)
    cmp = compare_reports(
        report_a, report_b, n_resamples=args.resamples,
        seed=args.seed, cohort=args.cohort)
    if args.as_json:
        print(json.dumps(cmp, indent=2, sort_keys=True))
    else:
        print(format_markdown(cmp, label_a=args.label_a, label_b=args.label_b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
