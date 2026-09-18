"""Span contract for the BERT intake path (mailroom-issues #85 M1/M3/#89).

Pure builders — the observability layer (Langfuse/Phoenix, the governed
constellation's ``intake-bert-prep`` mirror of ``intake-llm-prep``) consumes
these dicts; nothing in this module performs I/O so every span shape is
unit-testable without a tracer.  The contract surface is fixed by the epic:

- ``intake-bert-prep``  — the BERT compute path, mirror of intake-llm-prep;
- ``intake-ml-triage``   — the routing/triage span: input (filename, chars,
  windows), output (doc_type, subclass, confidence, route), metadata
  (model_id, artifact_sha, dataset_revision, calibration_version,
  label_schema_version, synthetic_policy_version, window_agreement).

Every ML dependency is wrapped by the caller (see ``mailroom_ml.inference``);
a failure is recorded via ``record_failure`` — the span carries the failure
reason and the pipeline continues on the LLM path (fail-open, plan D10).
"""
from __future__ import annotations

import traceback
from typing import Any

from mailroom_ml.config import INTAKE_HANDOFF_SCHEMA_VERSION

__all__ = [
    "SPAN_INTAKE_BERT_PREP",
    "SPAN_INTAKE_ML_TRIAGE",
    "build_bert_prep_span",
    "build_ml_triage_span",
    "build_section_map",
    "record_failure",
]

SPAN_INTAKE_BERT_PREP = "intake-bert-prep"
SPAN_INTAKE_ML_TRIAGE = "intake-ml-triage"


def build_section_map(
    sections: list[dict[str, Any]] | None,
    doc_chars: int,
    roles_catalog: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Validate + normalize a section map into the quality-bundle shape.

    Sections carry ``heading``, ``role``, ``start_offset``, ``end_offset``.
    The epic guard (P5): offsets in bounds, monotonic non-overlapping, roles
    within the catalog when one is given; short forms legitimately carry an
    empty map (``empty_by_design=True``).

    Returns ``{"ok": bool, "sections": [...], "guard_failures": [str...]}``.
    """
    roles_catalog = roles_catalog if roles_catalog is not None else frozenset()
    failures: list[str] = []
    clean: list[dict[str, Any]] = []
    if not sections:
        return {"ok": True, "sections": [], "guard_failures": []}
    prev_end = 0
    for i, s in enumerate(sections):
        heading = str(s.get("heading") or "")
        role = str(s.get("role") or "")
        try:
            start = int(s.get("start_offset", -1))
            end = int(s.get("end_offset", -1))
        except (TypeError, ValueError):
            failures.append(f"section[{i}] offsets not integers")
            continue
        if start < 0 or end < start or end > doc_chars:
            failures.append(
                f"section[{i}] offsets out of bounds ({start}..{end} / {doc_chars})")
            continue
        if start < prev_end:
            failures.append(f"section[{i}] overlaps previous section")
            continue
        if roles_catalog and role not in roles_catalog:
            failures.append(f"section[{i}] role {role!r} not in catalog")
            continue
        clean.append({"heading": heading, "role": role,
                      "start_offset": start, "end_offset": end})
        prev_end = end
    return {"ok": not failures, "sections": clean, "guard_failures": failures}


def build_bert_prep_span(
    *,
    filename: str,
    doc_chars: int,
    token_estimate: int,
    windows: int,
    model_id: str,
    artifact_sha: str | None,
    dataset_revision: str | None,
    calibration_version: str | None,
    label_schema_version: str | None,
    synthetic_policy_version: str | None,
    window_agreement: float | None,
    route: str | None,
    doc_type: str | None,
    subclass: str | None,
    confidence: float | None,
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``intake-bert-prep`` observation span (epic Phase 2 acceptance).

    Mirrors the fields the intake-llm-prep span already carries plus the
    model/artifact identity, so ops lamps can compare the two paths on the
    same axes (cost per doc, agreement, guard failures).
    """
    return {
        "name": SPAN_INTAKE_BERT_PREP,
        "input": {
            "filename": filename,
            "chars": doc_chars,
            "token_estimate": token_estimate,
            "windows": windows,
        },
        "output": {
            "doc_type": doc_type,
            "subclass": subclass,
            "confidence": confidence,
            "route": route,
            "window_agreement": window_agreement,
        },
        "metadata": {
            "model_id": model_id,
            "artifact_sha": artifact_sha,
            "dataset_revision": dataset_revision,
            "calibration_version": calibration_version,
            "label_schema_version": label_schema_version,
            "synthetic_policy_version": synthetic_policy_version,
        },
        "quality": quality or {},
    }


def build_ml_triage_span(
    *,
    schema_version: int = INTAKE_HANDOFF_SCHEMA_VERSION,
    method: str,
    routing_path: str,
    routing_reason: str,
    doc_chars: int,
    token_estimate: int,
    windows: int,
    model_id: str | None,
    artifact_sha: str | None,
    dataset_revision: str | None,
    calibration_version: str | None,
    label_schema_version: str | None,
    synthetic_policy_version: str | None,
    window_agreement: float | None,
    doc_type: str | None = None,
    subclass: str | None = None,
    confidence: float | None = None,
    route: str | None = None,
    guard_failures: list[str] | None = None,
) -> dict[str, Any]:
    """The ``intake-ml-triage`` span — the epic's M1/M3 observability surface.

    The span contract itemizes exactly: input (filename, chars, windows —
    passed here as scalar args), output (doc_type, subclass, confidence,
    route), metadata (model_id, artifact_sha, dataset_revision,
    calibration_version, label_schema_version, synthetic_policy_version,
    window_agreement).  Nothing here is optional from the contract's
    perspective; ``None`` values mean "not produced by this path".
    """
    return {
        "name": SPAN_INTAKE_ML_TRIAGE,
        "input": {
            "chars": doc_chars,
            "token_estimate": token_estimate,
            "windows": windows,
        },
        "output": {
            "doc_type": doc_type,
            "subclass": subclass,
            "confidence": confidence,
            "route": route,
            "window_agreement": window_agreement,
        },
        "metadata": {
            "schema_version": schema_version,
            "method": method,
            "routing_path": routing_path,
            "routing_reason": routing_reason,
            "model_id": model_id,
            "artifact_sha": artifact_sha,
            "dataset_revision": dataset_revision,
            "calibration_version": calibration_version,
            "label_schema_version": label_schema_version,
            "synthetic_policy_version": synthetic_policy_version,
        },
        "guard_failures": guard_failures or [],
    }


def record_failure(span: dict[str, Any], exc: BaseException, *, where: str) -> dict[str, Any]:
    """Attach a machine-readable failure record to a span (fail-open, D10).

    The trace keeps the exception type + message, the span-owner location
    and (when available) the first stack frame of the failure — never the
    full traceback text in the span payload (keeps payloads small and
    schema-stable).
    """
    return {
        **span,
        "failure": {
            "type": type(exc).__name__,
            "message": str(exc),
            "where": where,
            "traceback_point": traceback.format_exception_only(type(exc), exc)[-1].strip(),
        },
    }
