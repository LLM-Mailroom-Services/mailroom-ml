#!/usr/bin/env python3
"""Eval CLI for the ModernBERT intake classifier (plan §11, issues #92 M7).

Report-only harness (plan §11 surfaces 1-4):

    .venv/bin/python training/eval/eval_modernbert.py \\
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
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # training/<area>/<script>.py -> repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mailroom_ml.calibration import (  # noqa: E402
    ece_from_conf,
    ece_within_band,
    routing_thresholds_from_sweep,
    selective_risk_sweep,
    write_routing_thresholds,
)
from mailroom_ml.config import (  # noqa: E402
    CANONICAL_REPO,
    CANONICAL_REVISION,
    DATA_DIR,
    FAST_PATH_ERROR_BUDGET,
    FINETUNE_REPO,
    FINETUNE_REVISION,
    GATE_REQUIRED_AGREEMENT,
    HEAD_ECE_EXCLUSION_THRESHOLD,
    MAX_TOKENS,
    RANDOM_STATE,
    ROUTE_DOC_CONFIDENCE,
    ROUTE_MARGIN,
    ROUTE_SUBCLASS_CONFIDENCE,
    ROUTE_WINDOW_AGREEMENT,
    SELECTIVE_RISK_MIN_N,
    STAGE_DIR,
)
from mailroom_ml.inference import (  # noqa: E402
    BundleLoadError,
    BundleUnavailable,
    load_bundle,
)
from mailroom_ml.ood import score_ood  # noqa: E402
from mailroom_ml.windows import window_document  # noqa: E402

__all__ = [
    "build_comparable_metrics",
    "build_parser",
    "evaluate_documents",
    "format_report",
    "main",
    "stratified_sample",
]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="",
                    help="artifact bundle dir (default: ML_MODEL_DIR env or "
                         "artifacts/{pytorch,onnx}/model)")
    ap.add_argument("--stage", type=Path, default=STAGE_DIR,
                    help="staged tree with data/documents/test "
                         "(default data/modernbert_training/stage)")
    ap.add_argument("--subset", default="test",
                    choices=["test", "validation", "train", "all",
                             "heldout-plus"],
                    help="document pool: held-out test only, the trainer's "
                         "validation split (the calibration set routing "
                         "thresholds are fit on), train+val (no test), full "
                         "finetune corpus (all splits), or canonical test + "
                         "heldout-plus v1 extension "
                         "(training/dataset/mailroom-dataset/build_heldout_plus.py)")
    ap.add_argument("--sample", type=int, default=50,
                    help="per-doc_type stratified sample size (0 = all)")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--max-length", type=int, default=MAX_TOKENS)
    ap.add_argument("--selective-risk", action="store_true",
                    help="run the selective-risk threshold sweep and report "
                         "the deployment threshold")
    ap.add_argument("--write-routing-thresholds", type=Path, default=None,
                    help="write routing_thresholds.json (#25) from the "
                         "selective-risk sweep (implies --selective-risk). "
                         "PATH may be a file or a directory. Only allowed on "
                         "a pool without held-out test docs (--subset "
                         "validation / train): fitting the deployment "
                         "threshold on test would leak into every test "
                         "metric reported against it")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="print the report dict as JSON")
    ap.add_argument("--subclass-decode-logit-adjust", type=float, default=0.0,
                    help="add tau * log(train prior) to subclass logits at "
                         "decode (0 = shipped argmax). M9b tau_train=1.0: "
                         "0.5 yields effective tau 0.5 without retraining")
    ap.add_argument("--write-markdown", type=Path, default=None,
                    help="write TEST-EVAL markdown via training/eval/write_eval_report.py "
                         "(requires --run-tag)")
    ap.add_argument("--run-tag", default="",
                    help="run tag for --write-markdown output paths")
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


def _ensure_corpus_snapshot() -> None:
    """Fetch the pinned finetune corpus when ``data/parquet`` is absent (Modal)."""
    marker = DATA_DIR / "parquet" / "ground_truth" / "train"
    if marker.is_dir() and list(marker.glob("*.parquet")):
        return
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        print(
            f"[eval_modernbert] pulling {FINETUNE_REPO} @ {FINETUNE_REVISION} "
            f"-> {DATA_DIR}",
            file=sys.stderr,
            flush=True,
        )
        snapshot_download(
            repo_id=FINETUNE_REPO,
            repo_type="dataset",
            revision=FINETUNE_REVISION,
            local_dir=str(DATA_DIR),
        )
    except Exception as exc:
        if os.environ.get("HF_TOKEN"):
            raise
        print(
            f"[eval_modernbert] finetune pull failed ({exc}); "
            f"using public {CANONICAL_REPO} @ {CANONICAL_REVISION[:8]}…",
            file=sys.stderr,
            flush=True,
        )
        snapshot_download(
            repo_id=CANONICAL_REPO,
            repo_type="dataset",
            revision=CANONICAL_REVISION,
            local_dir=str(DATA_DIR),
            allow_patterns=["parquet/**"],
        )


def _load_parquet_split(stage: Path, *parts: str):
    import pandas as pd

    d = stage.joinpath(*parts)
    if d.is_dir() and list(d.glob("*.parquet")):
        return pd.concat(
            [pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))],
            ignore_index=True)
    return None


HELDOUT_PLUS_DIRNAME = "heldout_plus_v1"


def _heldout_plus_frame():
    """heldout-plus v1 extension rows (canonical test stays untouched)."""
    import pandas as pd

    candidates = [
        Path("/root/data") / HELDOUT_PLUS_DIRNAME / "documents.parquet",
        DATA_DIR / HELDOUT_PLUS_DIRNAME / "documents.parquet",
    ]
    for p in candidates:
        if p.is_file():
            return pd.read_parquet(p), p
    raise FileNotFoundError(
        "heldout-plus v1 absent: expected documents.parquet under "
        f"{candidates[1]} (build it with "
        "`uv run python training/dataset/mailroom-dataset/build_heldout_plus.py --help`)"
    )


# pools that contain held-out test documents: routing thresholds are never fit
# on these (plan §8: the deployment threshold comes from the calibration set)
THRESHOLD_FIT_REFUSED_SUBSETS = ("test", "all", "heldout-plus")


def _load_eval_docs(stage: Path, subset: str):
    """Load the eval document pool (test / validation / train / all / heldout-plus)."""
    if subset in ("test", "validation"):
        for parts in (
            ("parquet", "documents", subset),
            ("data", "documents", subset),
        ):
            frame = _load_parquet_split(stage, *parts)
            if frame is not None:
                return frame
        _ensure_corpus_snapshot()
        from mailroom_ml.dataset import build_documents, load_corpus_rows

        docs = build_documents(load_corpus_rows())
        return docs[docs["split"] == subset].reset_index(drop=True)

    _ensure_corpus_snapshot()
    from mailroom_ml.dataset import build_documents, load_corpus_rows

    docs = build_documents(load_corpus_rows())
    if subset == "train":
        return docs[docs["split"].isin(["train", "validation"])].reset_index(
            drop=True)
    if subset == "heldout-plus":
        import pandas as pd

        canon = docs[docs["split"] == "test"].reset_index(drop=True)
        plus, src = _heldout_plus_frame()
        overlap = set(canon["filename"].astype(str)) & set(
            plus["filename"].astype(str))
        if overlap:
            raise ValueError(
                f"heldout-plus overlaps canonical test on {len(overlap)} "
                f"filenames (e.g. {sorted(overlap)[:3]}); rebuild the "
                "extension before evaluating")
        # Align plus columns to the canonical frame (extra sidecars ride
        # along only when present in both).
        shared = [c for c in canon.columns if c in plus.columns]
        combined = pd.concat([canon[shared], plus[shared]],
                             ignore_index=True)
        combined.attrs["heldout_plus_source"] = str(src)
        return combined.reset_index(drop=True)
    return docs.reset_index(drop=True)


def evaluate_documents(bundle, docs, *, sample: int, seed: int,
                       max_length: int, selective_risk: bool = False,
                       doc_confidence: float | None = None,
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
    # fast_path_rate mirrors the production gate (classify_document): the
    # artifact routing overlay (#25) when present, else the config constants,
    # and BOTH the route and the plan's agreement floors.  An explicit
    # ``doc_confidence`` wins over the overlay.
    from mailroom_ml.inference import _bundle_route

    if doc_confidence is None:
        doc_confidence = _bundle_route(
            bundle, "ROUTE_DOC_CONFIDENCE", ROUTE_DOC_CONFIDENCE)
    gate = {
        "doc_confidence": float(doc_confidence),
        "subclass_confidence": _bundle_route(
            bundle, "ROUTE_SUBCLASS_CONFIDENCE", ROUTE_SUBCLASS_CONFIDENCE),
        "window_agreement": max(
            _bundle_route(bundle, "ROUTE_WINDOW_AGREEMENT",
                          ROUTE_WINDOW_AGREEMENT),
            GATE_REQUIRED_AGREEMENT),
        "margin": _bundle_route(bundle, "ROUTE_MARGIN", ROUTE_MARGIN),
    }
    filenames = docs["filename"].astype(str).tolist()
    strata = docs["doc_type"].astype(str).tolist()
    split_counts: dict[str, int] | None = None
    if "split" in docs.columns:
        split_counts = {
            str(k): int(v)
            for k, v in docs["split"].astype(str).value_counts().items()
        }
    if sample > 0:
        keep = stratified_sample(filenames, strata, sample, seed)
        docs = docs[docs["filename"].astype(str).isin(keep)].reset_index(drop=True)
        if "split" in docs.columns:
            split_counts = {
                str(k): int(v)
                for k, v in docs["split"].astype(str).value_counts().items()
            }

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
    per_doc: list[dict] = []
    n_fast = 0
    n_ood = 0
    n_ood_scored = 0
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
            version = getattr(bundle, "input_construction", "v1")
            filename = str(r.get("filename") or "")
            wins = window_document(
                title, doc_text, max_tokens=max_length,
                version=version, filename=filename)
        except (RuntimeError, OSError) as exc:  # transformers absent / tokenizer unloadable
            raise SystemExit(f"eval needs the transformers tokenizer: {exc}") from exc
        # window_document already applies the construction prefix — do not
        # re-decorate (that would double-prefix and mismatch training).
        decorated = wins
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
        dt_ok = dt_pred == gt_dt
        sc_ok = bool(dt_ok and sc_pred is not None and sc_pred == gt_sc)
        if dt_ok:
            correct_dt += 1
            cohorts[cohort]["correct_dt"] += 1
            dt_cond_denom += 1
            if gt_dt in subclass_heads:
                # conditional per-head macro-F1 input; a None sc_pred (abstain)
                # is a miss when a pair IS recorded.  The llm_overflow path
                # never reaches here (dt_pred != gt_dt): its docs stay in
                # support but are excluded from the macro-F1 denominator.
                per_head_pairs[gt_dt].append((gt_sc, sc_pred))
            if sc_ok:
                correct_sc += 1
        # gate on the unrounded values when classify_windows exposes them
        # (the reported fields are rounded to 4 dp: 0.969951 -> 0.97)
        gv = merged.get("gate_values") or merged
        p_dt = float(gv.get("calibrated_confidence") or 0.0)
        p_sc = float(gv.get("subclass_confidence") or 0.0)
        agree = float(gv.get("agreement") or 0.0)
        margin = float(gv.get("margin") or 0.0)
        fast = (
            dt_pred not in ("llm_overflow", "unknown")
            and p_dt >= gate["doc_confidence"]
            and (sc_pred is None or p_sc >= gate["subclass_confidence"])
            and agree >= gate["window_agreement"]
            and margin >= gate["margin"]
        )
        ood_flag = None
        logits = merged.get("window_doc_type_logits") or []
        if getattr(bundle, "ood_probe", None) and logits:
            _, ood_flag, _ = score_ood(np.stack(logits), bundle.ood_probe)
            n_ood_scored += 1
            if ood_flag:
                n_ood += 1
                fast = False
        if fast:
            n_fast += 1
        per_doc.append({
            "filename": str(r.get("filename") or ""),
            "gt_doc_type": gt_dt,
            "pred_doc_type": dt_pred,
            "dt_correct": dt_ok,
            "gt_subclass": gt_sc,
            "pred_subclass": sc_pred,
            "sc_correct": sc_ok,
            "n_windows": len(wins),
            "agreement": agree,
            "fast_path": fast,
            "ood_flag": ood_flag,
        })
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
        "fast_path_rate": round(n_fast / n, 4) if n else 0.0,
        "fast_path_gate": gate,
        "ood": {
            "probe_status": (
                "ok" if getattr(bundle, "ood_probe", None) else "absent"),
            "n_scored": n_ood_scored,
            "n_flagged": n_ood,
            "rate": round(n_ood / n_ood_scored, 4) if n_ood_scored else None,
        },
        "per_doc": per_doc,
        "recorded_gates": {
            "P0_doc_type": {"threshold": 0.95, "actual": round(acc_dt, 4),
                            "met": acc_dt >= 0.95, "report_only": True},
            "P0_subclass": {"threshold": 0.75, "actual": round(acc_sc, 4),
                            "met": acc_sc >= 0.75, "report_only": True},
        },
    }
    if split_counts is not None:
        report["eval_split_counts"] = split_counts
    if selective_risk and win_confs:
        report["selective_risk"] = _sweep_or_refuse(bundle, win_confs,
                                                    win_correct)
        for cohort, c in cohorts.items():
            if c["win_confs"]:
                report.setdefault("cohorts")[cohort]["selective_risk"] = \
                    _sweep_or_refuse(bundle, c["win_confs"], c["win_correct"])
    report["comparable_metrics"] = build_comparable_metrics(report)
    return report


def build_comparable_metrics(report: dict) -> dict:
    """Headline scalars for ``compare_runs.py`` / LLM sorter eval JSON pairing.

    Mirrors ``training/eval/compare_runs.py`` ``_SCALAR_KEYS`` plus per-head macro-F1,
    cohort agreement, and the #85 routing contract reference thresholds.
    """
    wc = report.get("window_calibration") or {}
    sr = report.get("selective_risk") or {}
    ood = report.get("ood") or {}
    per_doc = report.get("per_doc") or []
    agreements = [float(d["agreement"]) for d in per_doc if d.get("agreement") is not None]
    mean_agreement = round(float(np.mean(agreements)), 4) if agreements else None
    fast_path_docs = sum(1 for d in per_doc if d.get("fast_path"))
    per_head = report.get("per_head") or {}
    head_ece = report.get("head_ece") or {}
    cohorts_out: dict[str, dict] = {}
    for name, c in (report.get("cohorts") or {}).items():
        cohorts_out[name] = {
            "n_docs": c.get("n_docs"),
            "doc_type_accuracy": c.get("doc_type_accuracy"),
            "mean_agreement": c.get("mean_agreement"),
            "window_ece": c.get("window_ece"),
        }
        csr = c.get("selective_risk")
        if isinstance(csr, dict):
            cohorts_out[name]["selective_risk"] = {
                "refused": csr.get("refused"),
                "recommended_threshold": csr.get("recommended_threshold"),
                "budget_met": csr.get("budget_met"),
            }
    selective = None
    if sr:
        selective = {
            "refused": sr.get("refused"),
            "reason": sr.get("reason"),
            "recommended_threshold": sr.get("recommended_threshold"),
            "budget_met": sr.get("budget_met"),
            "coverage": sr.get("coverage"),
            "error_budget": sr.get("error_budget", FAST_PATH_ERROR_BUDGET),
            "n_at_pick": sr.get("n_at_pick"),
            "min_n": sr.get("min_n", SELECTIVE_RISK_MIN_N),
        }
    telemetry = report.get("run_telemetry") or {}
    return {
        "schema": "mailroom-ml/comparable-metrics/v1",
        "purpose": "paired compare_runs / LLM sorter shadow eval",
        "n_docs": report.get("n_docs"),
        "eval_subset": report.get("eval_subset"),
        "doc_type_accuracy": report.get("doc_type_accuracy"),
        "subclass_accuracy_conditional": report.get("subclass_accuracy_conditional"),
        "window_ece": wc.get("ece"),
        "window_band_ece": wc.get("band_ece"),
        "n_windows": wc.get("n_windows"),
        "mean_window_agreement": mean_agreement,
        "fast_path_rate": report.get("fast_path_rate"),
        "fast_path_n_docs": fast_path_docs,
        "selective_risk": selective,
        "ood_rate": ood.get("rate"),
        "ood_probe_status": ood.get("probe_status"),
        "per_head_macro_f1": {
            head: m.get("macro_f1") for head, m in sorted(per_head.items())
        },
        "per_head_ece": head_ece,
        "cohorts": cohorts_out,
        "routing_contract": {
            "composite_score": "p_calibrated * agreement * margin",
            "gate_thresholds": {
                "route_doc_confidence": ROUTE_DOC_CONFIDENCE,
                "route_subclass_confidence": ROUTE_SUBCLASS_CONFIDENCE,
                "route_window_agreement": ROUTE_WINDOW_AGREEMENT,
                "route_margin": ROUTE_MARGIN,
                "gate_required_agreement": GATE_REQUIRED_AGREEMENT,
            },
            "fast_path_error_budget": FAST_PATH_ERROR_BUDGET,
        },
        "latency_seconds_per_document": telemetry.get("latency_seconds_per_document"),
        "compare_runs_scalar_keys": [
            "doc_type_accuracy",
            "window_ece",
            "fast_path_rate",
            "selective_risk.recommended_threshold",
            "ood.rate",
        ],
    }


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
    merged["_window_probs"] = merged["window_doc_type_probs"]
    return merged


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
    lines += [
        "",
        f"fast-path rate    : {report.get('fast_path_rate')}",
        f"ood probe         : {(report.get('ood') or {}).get('probe_status')} "
        f"flagged={(report.get('ood') or {}).get('n_flagged')} "
        f"rate={(report.get('ood') or {}).get('rate')}",
    ]
    cm = report.get("comparable_metrics") or {}
    if cm:
        lines += ["", "comparable metrics (LLM sorter / compare_runs):"]
        lines.append(f"  schema            : {cm.get('schema')}")
        lines.append(f"  mean_agreement    : {cm.get('mean_window_agreement')}")
        sr_cm = cm.get("selective_risk") or {}
        if sr_cm.get("refused"):
            lines.append(f"  selective_risk    : REFUSED ({sr_cm.get('reason')})")
        elif sr_cm:
            lines.append(
                f"  selective_risk    : threshold={sr_cm.get('recommended_threshold')} "
                f"budget_met={sr_cm.get('budget_met')}")
    lines += ["", "recorded gates (report-only, plan §7):"]
    for name, g in report["recorded_gates"].items():
        lines.append(f"  {name:<12s} actual {g['actual']} vs "
                     f"threshold {g['threshold']} -> {'MET' if g['met'] else 'NOT MET'}"
                     f" (report_only={g['report_only']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.write_routing_thresholds is not None
            and args.subset in THRESHOLD_FIT_REFUSED_SUBSETS):
        raise SystemExit(
            f"--write-routing-thresholds refused on --subset {args.subset}: "
            "the pool contains held-out test docs, so the fitted threshold "
            "would leak into the test metrics. Fit it on the calibration set "
            "(--subset validation) and evaluate test separately.")
    try:
        bundle = load_bundle(args.checkpoint)
    except (BundleUnavailable, BundleLoadError) as exc:
        raise SystemExit(str(exc)) from exc
    bundle.subclass_decode_logit_adjust = args.subclass_decode_logit_adjust
    docs = _load_eval_docs(args.stage, args.subset)
    want_sweep = args.selective_risk or args.write_routing_thresholds
    report = evaluate_documents(
        bundle, docs, sample=args.sample, seed=args.seed,
        max_length=args.max_length, selective_risk=want_sweep)
    report["eval_subset"] = args.subset
    report["finetune_revision"] = FINETUNE_REVISION
    report["subclass_decode_logit_adjust"] = args.subclass_decode_logit_adjust
    if args.write_routing_thresholds is not None:
        sweep = report.get("selective_risk") or {
            "refused": True, "reason": "selective_risk not in report"}
        payload = routing_thresholds_from_sweep(sweep)
        payload["fit_subset"] = args.subset
        dest = write_routing_thresholds(args.write_routing_thresholds, payload)
        report["routing_thresholds_path"] = str(dest)
    if args.as_json:
        print(json.dumps(report, sort_keys=True, indent=2))
    else:
        print(format_report(report))
    if args.write_markdown is not None:
        if not args.run_tag:
            raise SystemExit("--write-markdown requires --run-tag")
        import subprocess
        import tempfile

        root = ROOT
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=root / "reports"
        ) as tmp:
            json.dump(report, tmp, sort_keys=True, indent=2)
            tmp_path = Path(tmp.name)
        kind = "heldout-plus" if args.subset == "heldout-plus" else "test"
        cmd = [
            sys.executable,
            str(root / "training" / "eval" / "write_eval_report.py"),
            kind,
            "--run-tag",
            args.run_tag,
            "--eval-json",
            str(tmp_path),
        ]
        if kind == "test":
            cmd.extend(["--write-training-report"])
        subprocess.run(cmd, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
