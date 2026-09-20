"""Calibration: per-head temperature scaling + selective-risk analysis (plan §8).

The deployment discipline (plan §8, issues #63/#92): a temperature-scaled
encoder probability is NOT automatically comparable to LLM confidence, so the
fast-path thresholds are whatever the selective-risk sweep on a calibration
set measures — never inherited constants.  This module provides the two
mathematical layers:

- ``fit_temperature`` / ``apply_temperature``: Platt-style per-head scaling
  (scipy ``minimize_scalar``, bounded), heads with < 2 rows or < 2 unique
  classes stay at T = 1.0;
- ``selective_risk_sweep``: a deterministic threshold sweep that builds
  reliability tables + ECE and picks the deployment threshold meeting the
  error budget (``FAST_PATH_ERROR_BUDGET``) with ECE within the 0.88–0.97
  band (``ece_within_band``) — the "calibration:classify"-shaped report the
  route gate consumes.

All inputs are numpy arrays (logits + integer labels) — no torch, no I/O —
so every function is unit-testable with synthetic logits.
"""
from __future__ import annotations

import functools
from typing import Any

import numpy as np

from mailroom_ml.config import FAST_PATH_ERROR_BUDGET

__all__ = [
    "fit_temperature",
    "apply_temperature",
    "softmax",
    "ece",
    "ece_from_conf",
    "ece_within_band",
    "reliability_table",
    "selective_risk_sweep",
]

_ECE_BAND_LO = 0.88
_ECE_BAND_HI = 0.97
_ECE_BANDS = 10
_T_MIN = 0.05
_T_MAX = 10.0


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax with optional temperature (stable, log-space)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Bounded temperature scaling (T minimizing NLL on validation logits).

    Mirrors the trainer's ``fit_temperature`` (mailroom-corpus-eda cf096fa)
    on numpy inputs; heads with < 2 rows or < 2 unique labels are left at
    T = 1.0 (uncalibratable — plan §8).
    """
    from scipy.optimize import minimize_scalar

    lg = np.asarray(logits, dtype=np.float64).reshape(len(logits), -1)
    lab = np.asarray(labels).astype(int).reshape(-1)
    if len(lab) < 2 or len(np.unique(lab)) < 2:
        return 1.0

    def nll(t: float) -> float:
        if t <= 1e-3:
            return 1e9
        z = lg / t
        z = z - z.max(axis=1, keepdims=True)
        log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -log_probs[np.arange(len(lab)), lab].mean()

    res = minimize_scalar(nll, bounds=(_T_MIN, _T_MAX), method="bounded")
    return float(res.x)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrated probabilities for logits under a fitted temperature."""
    return softmax(np.asarray(logits, dtype=np.float64), temperature)


def ece(logits: np.ndarray, labels: np.ndarray, n_bins: int = _ECE_BANDS) -> float:
    """Expected Calibration Error over equal-width confidence bins.

    Mirrors the trainer's torch ``ece`` on numpy inputs: confidence vs
    accuracy per bin, weighted by bin share; the top bin is closed on the
    right (``conf >= lo``).  Returns a float in [0, 1].
    """
    probs = softmax(logits)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(labels).astype(int)).astype(
        np.float64)
    return ece_from_conf(conf, correct, n_bins)


def ece_from_conf(conf: np.ndarray, correct: np.ndarray,
                  n_bins: int = _ECE_BANDS) -> float:
    """ECE from already-computed (confidence, correctness) pairs.

    The eval CLI measures window-level calibration from merged window
    probabilities (no logits at hand) — feeding those confidences into
    ``ece`` would softmax them a second time (an AxisError on 1-D input).
    This is the confidence-space entry point; ``ece`` delegates to it.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if len(conf) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        sel = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo)
        if sel.sum() == 0:
            continue
        total += (sel.sum() / n) * abs(correct[sel].mean() - conf[sel].mean())
    return float(total)


def ece_within_band(conf: np.ndarray, correct: np.ndarray,
                    lo: float = _ECE_BAND_LO, hi: float = _ECE_BAND_HI) -> float:
    """ECE restricted to the deployment confidence band (0.88–0.97, plan §8).

    The band is where the fast path lives; calibration quality OUTSIDE the
    band is irrelevant to the skip decision.  ``conf``/``correct`` may be
    per-window or per-document — the caller decides the unit.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    sel = (conf >= lo) & (conf <= hi)
    if sel.sum() == 0:
        return 0.0  # nothing in the band — vacuous, not an error
    c = conf[sel]
    ok = correct[sel]
    bins = np.linspace(lo, hi, _ECE_BANDS + 1)
    total = 0.0
    for i in range(_ECE_BANDS):
        b_lo, b_hi = bins[i], bins[i + 1]
        in_bin = (c >= b_lo) & (c <= b_hi) if i == _ECE_BANDS - 1 \
            else (c >= b_lo) & (c < b_hi)
        if in_bin.sum() == 0:
            continue
        total += (in_bin.sum() / len(c)) * abs(
            ok[in_bin].mean() - c[in_bin].mean())
    return float(total)


def reliability_table(conf: np.ndarray, correct: np.ndarray,
                      n_bins: int = _ECE_BANDS) -> list[dict[str, Any]]:
    """Reliability/calibration table rows (bin, count, avg_conf, accuracy).

    The shape `calibration:classify` reports: one row per confidence bin
    with observed frequency and empirical accuracy — the material for the
    reliability diagram (plan §8/§11).
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        sel = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo)
        n = int(sel.sum())
        rows.append({
            "bin": f"[{lo:.2f}, {hi:.2f})" if i < n_bins - 1 else f"[{lo:.2f}, {hi:.2f}]",
            "n": n,
            "avg_confidence": round(float(conf[sel].mean()), 4) if n else None,
            "accuracy": round(float(correct[sel].mean()), 4) if n else None,
        })
    return rows


def selective_risk_sweep(
    conf: np.ndarray,
    correct: np.ndarray,
    thresholds: list[float] | None = None,
    error_budget: float = FAST_PATH_ERROR_BUDGET,
    band_lo: float = _ECE_BAND_LO,
    band_hi: float = _ECE_BAND_HI,
) -> dict[str, Any]:
    """Threshold sweep: P(err | fast path) per threshold + deployment pick.

    For each candidate confidence threshold: the fast-path coverage (fraction
    at/above it), the conditional error rate among fast-pathed items
    (selective risk — plan §8: P(err | fast path) ≤ error budget) and the
    ECE within the deployment band.

    The recommended deployment threshold is the LOWEST threshold whose
    selective risk is within budget AND band ECE within 0.05 — lower
    thresholds maximize coverage, so the sweep picks the most permissive
    threshold that still meets the error budget (initial budget 0.02 per
    ``FAST_PATH_ERROR_BUDGET``).  Deterministic: thresholds sort ascending,
    ties prefer the higher threshold.

    Returns the report dict (``calibration:classify``-shaped):
    ``rows`` (per-threshold), ``recommended_threshold``, ``selective_risk``,
    ``band_ece``, ``coverage``, ``error_budget``, ``budget_met``.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.linspace(0.60, 0.99, 40)]
    thresholds = sorted(thresholds)  # ascending — deterministic
    if not thresholds:
        raise ValueError("selective_risk_sweep requires >= 1 threshold")

    rows: list[dict[str, Any]] = []
    for t in thresholds:
        sel = conf >= t
        n = int(sel.sum())
        risk = float(1.0 - correct[sel].mean()) if n else 1.0
        rows.append({
            "threshold": float(t),
            "coverage": float(n / len(conf)) if len(conf) else 0.0,
            "n": n,
            "selective_risk": risk,
            "band_ece": ece_within_band(conf[sel], correct[sel], band_lo, band_hi),
        })

    recommended: float | None = None
    for r in rows:  # ascending thresholds — first within budget is the pick
        if r["selective_risk"] <= error_budget and r["band_ece"] <= 0.05:
            recommended = r["threshold"]
            break

    best = rows[-1] if rows else {}
    return {
        "rows": rows,
        "recommended_threshold": recommended,
        "selective_risk": risk,
        "band_ece": best.get("band_ece", 0.0),
        "coverage": best.get("coverage", 0.0),
        "error_budget": error_budget,
        "budget_met": recommended is not None,
        "note": (
            "deployment threshold from selective risk on the calibration set "
            "(plan §8) — replaces config BERT_INTAKE_MIN_CONFIDENCE once "
            "this analysis lands"
        ),
    }


def _calibration_report_parts(
    logits_by_head: dict[str, np.ndarray],
    labels_by_head: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Shared report builder: per-head temperature + ECE (used by eval CLI)."""
    out: dict[str, Any] = {"heads": {}}
    for head in sorted(logits_by_head):
        lg = np.asarray(logits_by_head[head])
        lab = np.asarray(labels_by_head[head]).astype(int).reshape(-1)
        t = fit_temperature(lg, lab)
        probs = apply_temperature(lg, t)
        conf = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == lab).astype(np.float64)
        out["heads"][head] = {
            "temperature": round(t, 4),
            "n": int(len(lab)),
            "ece": round(ece(lg, lab), 4),
            "band_ece": round(ece_within_band(conf, correct), 4),
            "reliability": reliability_table(conf, correct),
        }
    return out


# Public alias used by eval CLI / calibration gate — keeps the head loop
# out of callers.  (Functools import kept for future memoized sweeps.)
_ = functools  # noqa: B018 — reserved for cached sweep results in eval CLI
