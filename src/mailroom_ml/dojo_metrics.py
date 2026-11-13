"""Eval-report metrics computed by the Dojo, the constellation's scoring core.

The eval harness hands its per-document rows here, and the numbers come from
``llm_dojo_scoring.ml.score_intake_model`` and the Dojo's ECE. The M9a gates read
them, so the same definitions score the classifier here, in llm-mailroom and in
the sandbox. Needs the ``dojo`` extra (``uv sync --extra dojo``; ``dev`` includes it).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from mailroom_ml.dojo_contract import DOJO_COMMIT


def _dojo():
    """Import the Dojo's intake scorer and ECE, with a clear error when absent."""
    try:
        from llm_dojo_scoring.analysis.calibration import expected_calibration_error
        from llm_dojo_scoring.ml import score_intake_model
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "eval metrics come from llm-dojo-scoring; install the dojo extra "
            "(uv sync --extra dojo) or the Modal eval image"
        ) from exc
    return score_intake_model, expected_calibration_error


def intake_records(per_doc: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Map eval ``per_doc`` rows onto the Dojo's intake record shape."""
    return [
        {
            "doc_type": row.get("gt_doc_type"),
            "subclass": row.get("gt_subclass") or None,
            "pred_doc_type": row.get("pred_doc_type"),
            "pred_subclass": row.get("pred_subclass"),
            "doc_type_confidence": row.get("doc_type_confidence"),
            "subclass_confidence": row.get("subclass_confidence"),
            "fast_path": bool(row.get("fast_path")),
        }
        for row in per_doc
    ]


def dojo_scorecard(
    per_doc: Iterable[Mapping[str, Any]],
    window_confs: Sequence[float] = (),
    window_correct: Sequence[bool] = (),
) -> dict[str, Any]:
    """The Dojo intake scorecard for one eval run, plus window-level ECE.

    ``window_ece`` is the Dojo's equal-width 10-bin ECE over window
    (confidence, correct) pairs; ``None`` when there are no windows.
    """
    score_intake_model, expected_calibration_error = _dojo()
    import llm_dojo_scoring

    card = score_intake_model(intake_records(per_doc))
    card["window_ece"] = expected_calibration_error(
        zip(
            (float(c) for c in window_confs),
            (bool(k) for k in window_correct),
            strict=True,
        )
    )
    card["dojo"] = {
        "commit": DOJO_COMMIT,
        "version": getattr(llm_dojo_scoring, "__version__", None),
    }
    return card
