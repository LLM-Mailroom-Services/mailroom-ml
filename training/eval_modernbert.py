#!/usr/bin/env python3
"""Eval CLI for the ModernBERT intake classifier (plan §11, issues #92 M7).

Report-only harness (plan §11 surfaces 1-4):

    .venv/bin/python training/eval_modernbert.py \\
        --checkpoint artifacts/onnx/model --subset test --sample 50 --seed 42

- **Data**: the held-out test split only (documents parquet from the staged
  tree ``data/modernbert_training/stage`` or ``--stage DIR``; the canonical
  corpus snapshot under ``data/parquet`` is the fallback).  The test split
  (323 docs) never touches training/calibration/threshold tuning (plan D11).
- **Stratified sampling**: ``--sample N --seed S`` draws a per-stratum
  (doc_type) proportional sample deterministically (RandomState, frozen
  algorithm — no sklearn).
- **Metrics**: doc_type + subclass accuracy, per-stratum confusion
  (plain-python counts), window-level ECE + band ECE + per-head temperature
  from the calibration module (plan §8), plus the selective-risk sweep
  report for the deployment threshold (``--selective-risk``).
- **Report-only**: gates are recorded, never enforced as an exit (the
  P0 thresholds doc_type >= 0.95 / subclass >= 0.75 belong to the eval
  harness, plan §7 test-gate note).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mailroom_ml.calibration import (  # noqa: E402
    apply_temperature,
    ece,
    ece_within_band,
    selective_risk_sweep,
)
from mailroom_ml.config import (  # noqa: E402
    RANDOM_STATE,
    ROUTE_DOC_CONFIDENCE,
    STAGE_DIR,
)
from mailroom_ml.inference import (  # noqa: E402
    BundleLoadError,
    BundleUnavailable,
    load_bundle,
    window_titles,
)
from mailroom_ml.windows import window_document  # noqa: E402

__all__ = ["build_parser", "stratified_sample", "evaluate_documents",
           "format_report", "main"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="",
                    help="artifact bundle dir (default: ML_MODEL_DIR env or "
                         "artifacts/{pytorch,onnx}/model)")
    ap.add_argument("--stage", type=Path, default=STAGE_DIR,
                    help="staged tree with parquet/documents/test "
                         "(default data/modernbert_training/stage)")
    ap.add_argument("--subset", default="test",
                    choices=["test"], help="eval split (held-out test only)")
    ap.add_argument("--sample", type=int, default=50,
                    help="per-doc_type stratified sample size (0 = all)")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--selective-risk", action="store_true",
                    help="run the selective-risk threshold sweep and report "
                         "the deployment threshold")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="emit the report as JSON")
    return ap


def stratified_sample(filenames, stratify: list[str], n: int, seed: int):
    """Deterministic per-stratum sample: min(n, stratum) from each class.

    Rows are sorted within their stratum, shuffled with the frozen
    ``np.random.RandomState(seed)`` algorithm (no sklearn), and the first
    ``min(n, len(stratum))`` rows per stratum are kept, in filename order
    for output stability.  Returns the selected filename list (subset of
    ``filenames`` in its original relative order).
    """
    rng = np.random.RandomState(seed)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for fn, s in zip(filenames, stratify, strict=True):
        by_stratum[s].append(fn)
    picked: set[str] = set()
    for s in sorted(by_stratum):
        rows = sorted(by_stratum[s])
        rng.shuffle(rows)
        picked.update(rows[:max(0, min(n, len(rows)))])
    return [fn for fn in filenames if fn in picked]


def _load_test_docs(stage: Path):
    """Held-out test documents from the staged tree (or corpus fallback)."""
    d = stage / "parquet" / "documents" / "test"
    if d.is_dir() and list(d.glob("*.parquet")):
        import pandas as pd

        return pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))],
                         ignore_index=True)
    # fallback: canonical corpus snapshot (load_corpus_rows verifies the pin)
    from mailroom_ml.dataset import build_documents, load_corpus_rows

    docs = build_documents(load_corpus_rows())
    return docs[docs["split"] == "test"]


def evaluate_documents(bundle, docs, *, sample: int, seed: int,
                       max_length: int, selective_risk: bool = False,
                       doc_confidence: float = ROUTE_DOC_CONFIDENCE,
                       ) -> dict:
    """Run the classifier over the sampled held-out test documents.

    Per document: window (title + body), plurality-vote merge via
    ``classify_windows``, compare vs the document label.  Window-level
    calibration: per-window doc_type confidence/correctness across all
    sampled windows feeds temperature fitting + ECE + (optionally) the
    selective-risk sweep — same unit as the plan's calibration surface.

    Returns the report dict; ``recorded_gates`` are report-only (never
    exits non-zero on accuracy).
    """
    filenames = docs["filename"].astype(str).tolist()
    strata = docs["doc_type"].astype(str).tolist()
    if sample > 0:
        keep = stratified_sample(filenames, strata, sample, seed)
        docs = docs[docs["filename"].astype(str).isin(keep)].reset_index(drop=True)

    rows = docs.to_dict("records")
    correct_dt = 0
    correct_sc = 0
    dt_cond_denom = 0
    confusion: dict[tuple[str, str], int] = Counter()
    win_confs: list[float] = []
    win_correct: list[bool] = []

    for r in rows:
        title = str(r.get("title") or "")
        doc_text = str(r.get("doc_text") or "")
        gt_dt = str(r["doc_type"])
        gt_sc = str(r.get("subclass") or "")
        try:
            wins = window_document(title, doc_text, max_tokens=max_length)
        except RuntimeError as exc:  # transformers absent
            raise SystemExit(f"eval needs the transformers tokenizer: {exc}") from exc
        decorated = window_titles(title, wins)
        try:
            merged = _merge_windows(bundle, decorated, max_length)
        except ValueError:
            # tokenizer drift: the windower clamps with the transformers
            # tokenizer, encode_inputs measures with the bundle's standalone
            # tokenizer — a window can measure a few tokens over budget.
            # Mirror production fail-open (classify_document routes LLM):
            # count the doc as an overflow miss, never crash the eval.
            merged = {"doc_type": "llm_overflow", "subclass": None,
                      "_window_probs": []}
        dt_pred = merged["doc_type"]
        sc_pred = merged["subclass"]
        confusion[(gt_dt, dt_pred)] += 1
        if dt_pred == gt_dt:
            correct_dt += 1
            dt_cond_denom += 1
            if sc_pred is not None and sc_pred == gt_sc:
                correct_sc += 1
        # window-level calibration data (doc_type head only)
        for p in merged["_window_probs"]:
            win_confs.append(float(p.max()))
            win_correct.append(bool(np.argmax(p) == _dt_id(bundle, gt_dt)))

    n = len(rows)
    acc_dt = correct_dt / n if n else 0.0
    acc_sc = correct_sc / dt_cond_denom if dt_cond_denom else 0.0

    report: dict = {
        "checkpoint": str(bundle.model_dir),
        "model_kind": bundle.model_kind,
        "artifact_sha": bundle.artifact_sha,
        "n_docs": n,
        "sample": sample,
        "seed": seed,
        "doc_type_accuracy": round(acc_dt, 4),
        "subclass_accuracy_conditional": round(acc_sc, 4),
        "per_stratum_confusion": {
            f"{gt}->{pred}": int(c)
            for (gt, pred), c in sorted(confusion.items())
        },
        "window_calibration": {
            "n_windows": len(win_confs),
            "ece": round(ece(np.asarray(win_confs), np.asarray(win_correct, dtype=int)), 4)
            if win_confs else None,
            "band_ece": round(ece_within_band(
                np.asarray(win_confs), np.asarray(win_correct, dtype=int)), 4)
            if win_confs else None,
        },
        "recorded_gates": {
            "P0_doc_type": {"threshold": 0.95, "actual": round(acc_dt, 4),
                            "met": acc_dt >= 0.95, "report_only": True},
            "P0_subclass": {"threshold": 0.75, "actual": round(acc_sc, 4),
                            "met": acc_sc >= 0.75, "report_only": True},
        },
    }
    if selective_risk and win_confs:
        report["selective_risk"] = selective_risk_sweep(
            np.asarray(win_confs), np.asarray(win_correct, dtype=int))
    return report


def _dt_id(bundle, doc_type: str) -> int:
    return int(bundle.maps["doc_type"]["label2id"].get(doc_type, -1))


def _merge_windows(bundle, decorated: list[str], max_length: int) -> dict:
    """Window plurality merge + per-window calibrated probs (eval-local)."""
    from mailroom_ml.inference import classify_windows

    merged = classify_windows(bundle, decorated, max_length=max_length)
    merged["_window_probs"] = _window_probs(bundle, decorated, max_length)
    return merged


def _window_probs(bundle, decorated: list[str], max_length: int) -> list[np.ndarray]:
    from mailroom_ml.inference import encode_inputs, predict

    ids, mask = encode_inputs(bundle, decorated, max_length=max_length)
    logits = predict(bundle, ids, mask)["doc_type"]
    t = bundle.temperatures.get("doc_type", 1.0)
    return [apply_temperature(row, t) for row in logits]


def format_report(report: dict) -> str:
    lines = [
        f"checkpoint        : {report['checkpoint']} ({report['model_kind']})",
        f"artifact_sha      : {report['artifact_sha']}",
        f"docs evaluated    : {report['n_docs']} (sample={report['sample']} "
        f"seed={report['seed']})",
        f"doc_type accuracy : {report['doc_type_accuracy']}",
        f"subclass accuracy : {report['subclass_accuracy_conditional']} "
        "(conditional on doc_type correct)",
        "",
        "per-stratum confusion (GT -> pred):",
    ]
    conf = report["per_stratum_confusion"]
    for pair in sorted(conf):
        lines.append(f"  {pair:<45s} {conf[pair]}")
    wc = report["window_calibration"]
    lines += [
        "",
        "window calibration (doc_type head):",
        f"  windows   : {wc['n_windows']}",
        f"  ece       : {wc['ece']}",
        f"  band ece  : {wc['band_ece']} (0.88-0.97 deployment band)",
    ]
    if report.get("selective_risk"):
        sr = report["selective_risk"]
        lines += [
            "",
            "selective risk (plan §8):",
            f"  recommended threshold : {sr['recommended_threshold']}",
            f"  budget met            : {sr['budget_met']} "
            f"(P(err|fast path) <= {sr['error_budget']})",
            f"  coverage at pick      : {sr['coverage']}",
        ]
    lines += ["", "recorded gates (report-only, plan §7):"]
    for name, g in report["recorded_gates"].items():
        lines.append(f"  {name:<12s} actual {g['actual']} vs "
                     f"threshold {g['threshold']} -> {'MET' if g['met'] else 'NOT MET'}"
                     f" (report_only={g['report_only']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bundle = load_bundle(args.checkpoint)
    except (BundleUnavailable, BundleLoadError) as exc:
        raise SystemExit(str(exc)) from exc
    docs = _load_test_docs(args.stage)
    report = evaluate_documents(
        bundle, docs, sample=args.sample, seed=args.seed,
        max_length=args.max_length, selective_risk=args.selective_risk)
    if args.as_json:
        print(json.dumps(report, sort_keys=True, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
