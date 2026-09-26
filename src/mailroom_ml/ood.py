"""OOD / novelty probe (#18) — energy score on doc_type logits.

The probe is CPU-servable (no extra encoder, no Hub download).  Fit on
in-distribution validation logits only (never the held-out test).  A
document is flagged OOD when its mean window energy exceeds the fitted
threshold; the fast-path gate then fails closed.

When no ``ood_probe.json`` sidecar is present the probe is ABSENT: we do
not invent a default that silently routes novel docs on the fast path.
Eval reports ``ood_probe_status`` so operators can see the gap.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mailroom_ml.config import (
    OOD_ENERGY_PERCENTILE,
    OOD_METHOD,
    OOD_PROBE_FILENAME,
    OOD_PROBE_TEMPERATURE,
)

__all__ = [
    "energy_score",
    "fit_ood_probe",
    "score_ood",
    "load_ood_probe",
    "write_ood_probe",
]


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Per-row energy ``-logsumexp(logits / T)``.  Higher = more unusual."""
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    t = float(temperature) if temperature else 1.0
    z = arr / t
    m = z.max(axis=-1, keepdims=True)
    logsumexp = m.squeeze(-1) + np.log(np.exp(z - m).sum(axis=-1))
    return -logsumexp


def fit_ood_probe(
    in_dist_logits: np.ndarray,
    *,
    percentile: float = OOD_ENERGY_PERCENTILE,
    temperature: float = OOD_PROBE_TEMPERATURE,
    method: str = OOD_METHOD,
) -> dict[str, Any]:
    """Fit a threshold from in-distribution validation logits.

    ``percentile`` is the tail mass treated as novel (default 5%): the
    threshold is the ``(100 - percentile)`` quantile of in-dist energy.
    A score *above* the threshold is OOD.  Calibration must use the
    validation split — never the held-out test (plan D11).
    """
    if method != "energy":
        raise ValueError(f"unsupported OOD method {method!r} (known: energy)")
    scores = energy_score(in_dist_logits, temperature)
    if scores.size == 0:
        raise ValueError("fit_ood_probe needs >= 1 in-distribution logit row")
    tail = max(0.0, min(50.0, float(percentile)))
    threshold = float(np.percentile(scores, 100.0 - tail))
    return {
        "method": "energy",
        "threshold": threshold,
        "temperature": float(temperature),
        "percentile": tail,
        "direction": "above",
        "n": int(scores.size),
        "note": (
            "Fit on validation logits only. Documents with mean window "
            "energy > threshold are OOD and fail the fast path."
        ),
    }


def score_ood(
    logits: np.ndarray,
    probe: dict[str, Any] | None,
) -> tuple[float | None, bool | None, str]:
    """Return ``(score, flag, status)``.

    ``status`` is ``absent`` when no probe is supplied (flag is None —
    callers must NOT treat that as "in-distribution").  ``ok`` when the
    probe scored the row.
    """
    if not probe:
        return None, None, "absent"
    scores = energy_score(logits, float(probe.get("temperature") or 1.0))
    score = float(np.mean(scores))
    threshold = float(probe["threshold"])
    flag = score > threshold
    return score, flag, "ok"


def load_ood_probe(model_dir: str | Path) -> dict[str, Any] | None:
    path = Path(model_dir) / OOD_PROBE_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict) or "threshold" not in data:
        return None
    return data


def write_ood_probe(path: str | Path, probe: dict[str, Any]) -> Path:
    dest = Path(path)
    if dest.is_dir() or dest.suffix != ".json":
        dest = dest / OOD_PROBE_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return dest
