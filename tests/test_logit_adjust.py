"""Hermetic tests for Menon log-prior + decode residual (mailroom-issues #112)."""
from __future__ import annotations

import numpy as np
import pytest

from mailroom_ml.logit_adjust import apply_logit_adjust, log_prior_values


def test_log_prior_inverts_inverse_frequency_weights():
    weights = {"a": 40 / (2 * 30), "b": 40 / (2 * 10)}
    lp = np.array(log_prior_values(weights, ["a", "b", "unseen"]))
    p = np.exp(lp)
    assert p[0] / p[1] == pytest.approx(3.0)
    assert p[2] == pytest.approx(p[1])
    assert p.sum() == pytest.approx(1.0)


def test_apply_logit_adjust_zero_is_noop():
    lg = np.array([[1.0, 2.0]], dtype=np.float32)
    out = apply_logit_adjust(lg, {"x": 1.0, "y": 2.0}, ["x", "y"], 0.0)
    assert out is lg


def test_apply_logit_adjust_majority_wins_from_ties():
    weights = {"email": 0.28, "attorney_demand": 80.3}
    lg = np.zeros((1, 2), dtype=np.float32)
    out = apply_logit_adjust(lg, weights, ["email", "attorney_demand"], 1.0)
    assert int(out[0].argmax()) == 0
