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
    wilson_lower,
)
from mailroom_ml.config import FAST_PATH_ERROR_BUDGET


def test_softmax_normalizes_and_within_range():
    lg = np.array([[1.0, 2.0, 3.0]])
    p = softmax(lg)
    assert p.sum() == pytest.approx(1.0)
    assert softmax(lg * 3)[0].argmax() == 2
    with pytest.raises(ValueError):
        softmax(lg, temperature=0.0)


def test_softmax_accepts_single_row():
    """#104 regression: the eval harness feeds 1-D per-window probability
    rows through apply_temperature — 1-D input must not AxisError."""
    row = np.array([0.0, 0.0, 10.0])
    p = softmax(row)
    assert p.shape == (3,)
    assert p.sum() == pytest.approx(1.0)
    assert p.argmax() == 2
    assert softmax(row, temperature=2.0).shape == (3,)


def test_fit_temperature_recovers_known_scale():
    pytest.importorskip("scipy")
    rng = np.random.RandomState(0)
    z = rng.normal(size=(4000, 5))
    t_true = 1.7
    p = np.exp(z / t_true)
    p /= p.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(5, p=p[i]) for i in range(len(p))])
    t_hat = fit_temperature(z, labels)
    assert abs(t_hat - t_true) < 0.25, t_hat


def test_fit_temperature_leaves_degenerate_heads_at_one():
    pytest.importorskip("scipy")
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
    """Confidence/accuracy where the Wilson risk clears the budget above
    0.97 (errors concentrated below 0.965) — the lowest safe threshold is
    the pick (maximize coverage)."""
    rng = np.random.RandomState(42)
    conf = np.clip(rng.normal(0.97, 0.01, 2000), 0.0, 1.0)
    correct = (conf > 0.965).astype(float)

    a = selective_risk_sweep(conf, correct)
    b = selective_risk_sweep(conf, correct)
    assert a == b  # deterministic

    assert a["recommended_threshold"] == 0.97  # lowest candidate
    assert a["n_at_pick"] >= a["min_n"]
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
                        "min_n", "n_at_pick", "insufficient_data", "note"}
    assert isinstance(res["rows"], list)
    assert all("threshold" in r and "coverage" in r and
               "selective_risk" in r and "band_ece" in r for r in res["rows"])


# ---------------------------------------------------------------------------
# #104 min-n guard + Wilson lower bound
# ---------------------------------------------------------------------------

def test_wilson_lower_hand_computed():
    """Wilson score lower bound: n=100, acc=0.9 -> ~0.825 (z=1.96)."""
    lo = wilson_lower(0.9, 100)
    assert lo == pytest.approx(0.8254, abs=1e-3)
    # degenerate inputs
    assert wilson_lower(0.0, 0) == 0.0
    assert wilson_lower(1.0, 1) > 0.0  # n=1 still bounded away from 1.0
    assert 0.0 <= wilson_lower(0.5, 30) <= 0.5


def test_sweep_never_recommends_from_tiny_n():
    """#104: 1 correct doc at high confidence must NOT yield budget_met."""
    conf = np.array([0.99, 0.98, 0.97, 0.96])
    correct = np.array([1, 1, 1, 1])  # 100% accurate, but n=4 < min_n
    res = selective_risk_sweep(conf, correct)
    assert res["budget_met"] is False
    assert res["recommended_threshold"] is None
    assert res["insufficient_data"] is True
    assert res["n_at_pick"] is None
    # every row carries the Wilson bound, not the raw rate
    for r in res["rows"]:
        assert "wilson_lower_acc" in r and "accuracy" in r


def test_sweep_recommends_only_with_min_n_met():
    """Same accuracy profile but n >= min_n -> a pick exists."""
    rng = np.random.RandomState(11)
    conf = np.clip(rng.normal(0.985, 0.005, 400), 0.0, 1.0)
    correct = (conf > 0.975).astype(float)
    res = selective_risk_sweep(conf, correct)
    assert res["insufficient_data"] is False
    assert res["recommended_threshold"] == 0.98
    assert res["n_at_pick"] >= res["min_n"]
    picked = [r for r in res["rows"]
              if r["threshold"] == res["recommended_threshold"]][0]
    assert picked["n"] >= res["min_n"]
    assert picked["selective_risk"] <= FAST_PATH_ERROR_BUDGET


def test_sweep_wilson_risk_is_conservative_vs_raw():
    """The Wilson error rate is >= the raw error rate at every threshold."""
    rng = np.random.RandomState(5)
    conf = rng.uniform(0.6, 1.0, 500)
    correct = (rng.uniform(size=500) < conf).astype(float)
    res = selective_risk_sweep(conf, correct)
    for r in res["rows"]:
        raw = 1.0 - r["accuracy"]
        assert r["selective_risk"] >= raw - 1e-9
