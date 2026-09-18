"""Calibration + selective-risk tests (plan §8, issues #63/#92).

Hermetic math over synthetic logits: temperature recovery, ECE
hand-computations, band ECE, reliability tables and the deterministic
selective-risk sweep (deployment threshold = lowest meeting the error
budget; ties prefer the higher threshold).
"""
from __future__ import annotations

import numpy as np
import pytest

from mailroom_ml.calibration import (
    apply_temperature,
    ece,
    ece_within_band,
    fit_temperature,
    reliability_table,
    selective_risk_sweep,
    softmax,
)
from mailroom_ml.config import FAST_PATH_ERROR_BUDGET


def test_softmax_normalizes_and_within_range():
    lg = np.array([[1.0, 2.0, 3.0]])
    p = softmax(lg)
    assert p.sum() == pytest.approx(1.0)
    assert softmax(lg * 3)[0].argmax() == 2
    with pytest.raises(ValueError):
        softmax(lg, temperature=0)


def test_fit_temperature_recovers_known_scale():
    rng = np.random.RandomState(0)
    z = rng.normal(size=(4000, 5))
    t_true = 1.7
    p = np.exp(z / t_true)
    p /= p.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(5, p=p[i]) for i in range(len(p))])
    t_hat = fit_temperature(z, labels)
    assert abs(t_hat - t_true) < 0.25, t_hat


def test_fit_temperature_leaves_degenerate_heads_at_one():
    assert fit_temperature(np.zeros((3, 3)), np.zeros(3)) == 1.0
    assert fit_temperature(np.zeros((1, 3)), np.array([0])) == 1.0


def test_ece_hand_computed():
    lg = np.array([[10.0, 0.0], [0.0, 10.0]])
    assert ece(lg, np.array([0, 1])) == pytest.approx(0.0, abs=1e-4)
    lg2 = np.array([[10.0, 0.0], [10.0, 0.0]])
    assert ece(lg2, np.array([0, 1])) == pytest.approx(0.5, abs=1e-3)
    lg3 = np.array([[1.0, 0.0]] * 4)
    assert ece(lg3, np.array([0, 0, 0, 1])) == pytest.approx(
        abs(0.75 - 1 / (1 + np.exp(-1))))


def test_apply_temperature_changes_confidence():
    lg = np.array([[3.0, 1.0], [0.5, 0.2]], dtype=float)
    p1 = apply_temperature(lg, 1.0)
    p2 = apply_temperature(lg, 3.0)
    assert p1.max() > p2.max()  # higher temperature flattens confidence
    assert p1.argmax(axis=1).tolist() == p2.argmax(axis=1).tolist()


def test_ece_within_band_ignores_outside_mass():
    conf = np.array([0.96, 0.95, 0.50, 0.10])   # first two inside the band
    correct = np.array([1.0, 0.0, 1.0, 0.0])
    in_band = ece_within_band(conf, correct)
    # the 0.95 "wrong" dominates the band -> nonzero ECE
    assert in_band > 0
    # hand-computed: only the two in-band windows count; each lands in its
    # own 0.009-wide bin, weighted 0.5 each:
    #   |1.0 - 0.96|*0.5 + |0.0 - 0.95|*0.5 = 0.02 + 0.475
    mid = ece_within_band(conf, np.array([1.0, 0.0, 1.0, 1.0]))
    assert mid == pytest.approx(0.5 * (1 - 0.96) + 0.5 * 0.95, abs=1e-4)
    # everything outside the band -> vacuous zero (not an error)
    assert ece_within_band(np.array([0.2]), np.array([1.0])) == 0.0


def test_reliability_table_rows_well_formed():
    rng = np.random.RandomState(1)
    conf = rng.uniform(0.5, 1.0, 500)
    correct = (rng.uniform(size=500) < conf).astype(float)
    rows = reliability_table(conf, correct, n_bins=5)
    assert len(rows) == 5
    assert sum(r["n"] for r in rows) == 500
    populated = [r for r in rows if r["n"]]
    for r in populated:
        assert 0.0 <= r["accuracy"] <= 1.0
        assert 0.0 <= r["avg_confidence"] <= 1.0


def test_selective_risk_sweep_deterministic_and_picks_lowest_safe():
    """Confidence/accuracy where risk clears the budget above 0.90."""
    rng = np.random.RandomState(42)
    conf = np.clip(rng.normal(0.9, 0.06, 2000), 0.0, 1.0)
    correct = (rng.uniform(size=2000) < np.clip(conf - 0.02, 0, 1)).astype(float)

    a = selective_risk_sweep(conf, correct)
    b = selective_risk_sweep(conf, correct)
    assert a == b  # deterministic

    picked = [r for r in a["rows"] if r["threshold"] == a["recommended_threshold"]]
    assert len(picked) == 1
    assert picked[0]["selective_risk"] <= FAST_PATH_ERROR_BUDGET
    # the lowest safe threshold is the pick (maximize coverage)
    lower = [r for r in a["rows"] if r["threshold"] < a["recommended_threshold"]]
    for r in lower:
        assert r["selective_risk"] > FAST_PATH_ERROR_BUDGET


def test_selective_risk_sweep_budget_unmet():
    """Perfectly random confidence (accuracy 0.5 at every level) -> no pick."""
    rng = np.random.RandomState(7)
    conf = rng.uniform(0.0, 1.0, 500)
    correct = (rng.uniform(size=500) < 0.5).astype(float)
    res = selective_risk_sweep(conf, correct)
    assert res["budget_met"] is False
    assert res["recommended_threshold"] is None


def test_selective_risk_sweep_returns_report_shape():
    rng = np.random.RandomState(3)
    conf = rng.uniform(0.5, 1.0, 300)
    correct = (rng.uniform(size=300) < conf).astype(float)
    res = selective_risk_sweep(conf, correct)
    assert set(res) == {"rows", "recommended_threshold", "selective_risk",
                        "band_ece", "coverage", "error_budget", "budget_met",
                        "note"}
    assert isinstance(res["rows"], list)
    assert all("threshold" in r and "coverage" in r and
               "selective_risk" in r and "band_ece" in r for r in res["rows"])
