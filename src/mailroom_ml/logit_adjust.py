"""Logit-adjusted classification (Menon et al. 2021) — shared train/decode.

Train-time (``LossConfig.subclass_logit_adjust``): add ``tau * log(pi)`` to
subclass logits **inside the loss only**. Rare classes need a larger raw
margin to win during CE, so unadjusted inference logits are rare-boosted.

Decode-time (``ModelBundle.subclass_decode_logit_adjust``): add the same
term to **raw inference logits** before temperature/argmax. For a checkpoint
trained with ``tau_train``, decode delta ``d`` yields effective

    tau_eff = tau_train - d

because ``argmax(raw + d log pi) = argmax(logits_true - (tau_train - d) log pi)``.
M9b (``20261006-021245``) used ``tau_train=1.0``; ``d=0.5`` is effective
tau 0.5 without a new GPU train. ``d=0`` (default) preserves shipped argmax.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

__all__ = ["log_prior_values", "apply_logit_adjust"]


def log_prior_values(weights: Mapping[str, float],
                     labels: Sequence[str]) -> list[float]:
    """Per-label log train prior from inverse-frequency weights.

    ``class_weights`` stores ``N / (K * n_k)``, so ``n_k ∝ 1 / w_k``. A label
    with no stored weight has no train support; it gets the smallest prior
    seen, so the adjustment never favours it.
    """
    inv = [1.0 / weights[lab] for lab in labels if weights.get(lab)]
    floor = min(inv) if inv else 1.0
    raw = [1.0 / weights[lab] if weights.get(lab) else floor for lab in labels]
    total = sum(raw) or 1.0
    return [float(np.log(v / total)) for v in raw]


def apply_logit_adjust(logits: np.ndarray, weights: Mapping[str, float],
                       labels: Sequence[str], tau: float) -> np.ndarray:
    """``logits + tau * log(pi)`` along the class axis (last dim).

    ``labels`` must match the logit width (trainable head order). ``tau=0``
    returns ``logits`` unchanged (no copy).
    """
    if not tau:
        return logits
    prior = np.asarray(log_prior_values(weights, labels), dtype=np.float32)
    if prior.shape[-1] != logits.shape[-1]:
        n = min(int(prior.shape[-1]), int(logits.shape[-1]))
        prior = prior[:n]
        logits = np.array(logits, copy=True)
        logits[..., :n] = logits[..., :n] + tau * prior
        return logits
    return logits + tau * prior
