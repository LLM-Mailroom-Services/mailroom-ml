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
  from the calibration module (plan §8), per-head document-level test
  macro-F1 + per-class support for the subclass heads (#112 M9a-U4), plus the
  selective-risk sweep report for the deployment threshold
  (``--selective-risk``).
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
    ece_from_conf,
    ece_within_band,
    selective_risk_sweep,
)
from mailroom_ml.config import (  # noqa: E402
    HEAD_ECE_EXCLUSION_THRESHOLD,
    RANDOM_STATE,
    ROUTE_DOC_CONFIDENCE,
    SELECTIVE_RISK_MIN_N,
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
                    help="staged tree with data/documents/test "
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
    for d in (stage / "parquet" / "documents" / "test",   # legacy EDA layout
              stage / "data" / "documents" / "test"):     # build_dataset layout
        if d.is_dir() and list(d.glob("*.parquet")):
            import pandas as pd

            return pd.concat(
                [pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))],
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
    exits non-zero on accuracy).  Per-head macro-F1 is conditional on the
    doc_type being correct: a doc whose merged doc_type is ``llm_overflow``
    still contributes to its head's per-class ``support``, but records no
    ``(gt, pred)`` pair, so it is excluded from the macro-F1 denominator.
    """
    filenames = docs["filename"].astype(str).tolist()
    strata = docs["doc_type"].astype(str).tolist()
    if sample > 0:
        keep = stratified_sample(filenames, strata, sample, seed)
        docs = docs[docs["filename"].astype(str).isin(keep)].reset_index(drop=True)

    # #112 M9a-U4: the per-head test macro-F1 surface.  The subclass head
    # names are the doc_type labels minus the inference-only ``unknown`` (the
    # data-driven source of truth — never a hard-coded list).
    doc_type_labels = (bundle.maps.get("doc_type", {}).get("labels")
                       or sorted(bundle.maps.get("doc_type", {})
                                 .get("label2id", {})))
    subclass_heads = [c for c in doc_type_labels if c != "unknown"]
    # document-level (gt_sc, sc_pred) pairs per head, restricted to docs whose
    # doc_type is correct for that head's class — the same conditional
    # convention as ``subclass_accuracy_conditional`` below.
    per_head_pairs: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    # per-head per-class support over ALL test docs of that head's doc_type
    # (the interpretability denominator; zero-support classes are surfaced).
    per_head_support: dict[str, Counter] = defaultdict(Counter)

    rows = docs.to_dict("records")
    correct_dt = 0
    correct_sc = 0
    dt_cond_denom = 0
    confusion: dict[tuple[str, str], int] = Counter()
    win_confs: list[float] = []
    win_correct: list[bool] = []
    # #104 cohort split: single-window (agreement trivially 1.0 — the gate
    # reduces to one miscalibrated p) vs multi-window — scored separately
    # so the cohorts never conflate.
    cohorts: dict[str, dict] = {
        "single-window": {"n_docs": 0, "correct_dt": 0, "agreements": [],
                          "win_confs": [], "win_correct": []},
        "multi-window": {"n_docs": 0, "correct_dt": 0, "agreements": [],
                         "win_confs": [], "win_correct": []},
    }

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
            # the doc still counts in its head's per-class support below, but
            # records no conditional (gt, pred) pair, so it is excluded from
            # the per-head macro-F1 denominator rather than scored as a miss.
            merged = {"doc_type": "llm_overflow", "subclass": None,
                      "_window_probs": []}
        dt_pred = merged["doc_type"]
        sc_pred = merged["subclass"]
        confusion[(gt_dt, dt_pred)] += 1
        if gt_dt in subclass_heads:
            # support is unconditional on the prediction: every test doc of
            # this head's doc_type contributes to its per-class support.
            per_head_support[gt_dt][gt_sc] += 1
        cohort = "single-window" if len(wins) == 1 else "multi-window"
        cohorts[cohort]["n_docs"] += 1
        cohorts[cohort]["agreements"].append(float(merged.get("agreement", 0.0)))
        if dt_pred == gt_dt:
            correct_dt += 1
            cohorts[cohort]["correct_dt"] += 1
            dt_cond_denom += 1
            if gt_dt in subclass_heads:
                # conditional per-head macro-F1 input; a None sc_pred (abstain)
                # is a miss when a pair IS recorded.  The llm_overflow path
                # never reaches here (dt_pred != gt_dt): its docs stay in
                # support but are excluded from the macro-F1 denominator.
                per_head_pairs[gt_dt].append((gt_sc, sc_pred))
            if sc_pred is not None and sc_pred == gt_sc:
                correct_sc += 1
        # window-level calibration data (doc_type head only)
        for p in merged["_window_probs"]:
            win_confs.append(float(p.max()))
            win_correct.append(bool(np.argmax(p) == _dt_id(bundle, gt_dt)))
            cohorts[cohort]["win_confs"].append(float(p.max()))
            cohorts[cohort]["win_correct"].append(
                bool(np.argmax(p) == _dt_id(bundle, gt_dt)))

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
            # confidence-space ECE: win_confs are already max-probabilities,
            # not logits — ece() would softmax them a second time (AxisError)
            "ece": round(ece_from_conf(
                np.asarray(win_confs), np.asarray(win_correct, dtype=int)), 4)
            if win_confs else None,
            "band_ece": round(ece_within_band(
                np.asarray(win_confs), np.asarray(win_correct, dtype=int)), 4)
            if win_confs else None,
        },
        "cohorts": _cohort_report(cohorts),
        "head_ece": _head_ece_report(bundle),
        "per_head": _per_head_report(bundle, subclass_heads, per_head_pairs,
                                     per_head_support),
        "recorded_gates": {
            "P0_doc_type": {"threshold": 0.95, "actual": round(acc_dt, 4),
                            "met": acc_dt >= 0.95, "report_only": True},
            "P0_subclass": {"threshold": 0.75, "actual": round(acc_sc, 4),
                            "met": acc_sc >= 0.75, "report_only": True},
        },
    }
    if selective_risk and win_confs:
        report["selective_risk"] = _sweep_or_refuse(bundle, win_confs,
                                                    win_correct)
        for cohort, c in cohorts.items():
            if c["win_confs"]:
                report.setdefault("cohorts")[cohort]["selective_risk"] = \
                    _sweep_or_refuse(bundle, c["win_confs"], c["win_correct"])
    return report


def _cohort_report(cohorts: dict[str, dict]) -> dict:
    """#104 per-cohort stats: n, accuracy, mean agreement, window ECE."""
    out: dict[str, dict] = {}
    for name, c in cohorts.items():
        n = c["n_docs"]
        wc = np.asarray(c["win_confs"])
        wk = np.asarray(c["win_correct"], dtype=int)
        out[name] = {
            "n_docs": n,
            "doc_type_accuracy": round(c["correct_dt"] / n, 4) if n else None,
            "mean_agreement": round(float(np.mean(c["agreements"])), 4)
            if c["agreements"] else None,
            "window_ece": round(ece_from_conf(wc, wk), 4) if len(wc) else None,
        }
    return out


def _head_ece_report(bundle) -> dict:
    """#104 per-head ECE sidecar surfaced from the bundle's exclusion policy.

    Pre-#107 artifacts carry no sidecar -> every head reports ``None`` and
    threshold passes are refused (see ``_sweep_or_refuse``).
    """
    policy = bundle.exclusion_policy or {}
    excluded = policy.get("excluded") or {}
    return {
        name: (info.get("ece_calibrated") if isinstance(info, dict) else None)
        for name, info in sorted(excluded.items())
    }


def _macro_f1_observed(pairs: list[tuple[str, str | None]]) -> float | None:
    """Macro-F1 over observed GT classes from document-level (gt, pred) pairs.

    Mirrors the trainer's ``macro_f1(..., observed_only=True)``: zero-row
    classes are excluded from the average, and a ``None`` prediction (the
    llm_overflow / abstain path) is a miss.  Returns ``None`` when there are
    no documents (an unmeasurable head, never a fabricated 0.0).
    """
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
    return float(np.mean(f1s))


def _per_head_report(bundle, subclass_heads: list[str],
                     pairs_by_head: dict[str, list], support_by_head: dict,
                     ) -> dict:
    """#112 M9a-U4 per-head test macro-F1 + per-class support.

    Every subclass head is reported (not just the predicted class's head):
    ``macro_f1`` is the document-level conditional macro-F1 (docs whose
    ``dt_pred == gt_dt`` for that head's class), ``support`` the per-class
    count of held-out test docs of that doc_type (zero-support classes are
    surfaced so a macro-F1 is interpretable).  A head with no conditional
    docs reports ``macro_f1: null``.
    """
    out: dict[str, dict] = {}
    for head in subclass_heads:
        labels = (bundle.maps.get(head) or {}).get("labels") or []
        support = {str(lab): 0 for lab in labels}
        for lab, count in (support_by_head.get(head) or {}).items():
            support[str(lab)] = int(count)
        mf1 = _macro_f1_observed(pairs_by_head.get(head, []))
        out[head] = {
            "macro_f1": round(mf1, 4) if mf1 is not None else None,
            "support": support,
        }
    return out


def _sweep_or_refuse(bundle, confs: list[float], correct: list[bool]) -> dict:
    """#104: run the min-n + Wilson sweep, or refuse when the doc_type head
    has no shipped per-head ECE or its ECE >= HEAD_ECE_EXCLUSION_THRESHOLD
    (uncalibratable heads must not pass thresholds)."""
    policy = bundle.exclusion_policy or {}
    excluded = policy.get("excluded") or {}
    dt_info = excluded.get("doc_type")
    ece_val = dt_info.get("ece_calibrated") if isinstance(dt_info, dict) else None
    if ece_val is None:
        return {"refused": True,
                "reason": "no per-head ECE sidecar in the artifact "
                          "(pre-#107 publish)"}
    if ece_val >= HEAD_ECE_EXCLUSION_THRESHOLD:
        return {"refused": True,
                "reason": f"doc_type ECE {ece_val} >= "
                          f"{HEAD_ECE_EXCLUSION_THRESHOLD} (uncalibratable)"}
    sweep = selective_risk_sweep(
        np.asarray(confs), np.asarray(correct, dtype=int),
        min_n=SELECTIVE_RISK_MIN_N)
    sweep["refused"] = False
    return sweep


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
    lines += ["", "cohorts (#104 single- vs multi-window):"]
    for name, c in report["cohorts"].items():
        lines.append(
            f"  {name:<14s} n={c['n_docs']:<4d} acc={c['doc_type_accuracy']} "
            f"mean_agreement={c['mean_agreement']} window_ece={c['window_ece']}")
    lines += ["", "per-head ECE sidecar (#104):"]
    for head, ece_val in report["head_ece"].items():
        lines.append(f"  {head:<20s} {ece_val}")
    lines += ["", "per-head test macro-F1 (#112 M9a-U4):"]
    for head, m in report["per_head"].items():
        support = m["support"]
        n_obs = sum(1 for c in support.values() if c > 0)
        lines.append(f"  {head:<20s} macro_f1={m['macro_f1']} "
                     f"(classes={n_obs}/{len(support)})")
        for cls in sorted(support):
            lines.append(f"      {cls:<26s} n={support[cls]}")
    if report.get("selective_risk"):
        sr = report["selective_risk"]
        lines += [
            "",
            "selective risk (plan §8, #104 min-n + Wilson):",
        ]
        if sr.get("refused"):
            lines.append(f"  REFUSED: {sr['reason']}")
        else:
            lines += [
                f"  recommended threshold : {sr['recommended_threshold']}",
                f"  budget met            : {sr['budget_met']} "
                f"(P(err|fast path) <= {sr['error_budget']})",
                f"  coverage at pick      : {sr['coverage']} "
                f"(n={sr['n_at_pick']}, min_n={sr['min_n']})",
                f"  insufficient data     : {sr['insufficient_data']}",
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
