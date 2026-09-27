"""Modal ModernBERT eval — wall time and GPU cost (sandbox parity)."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

L4_USD_PER_HOUR = 0.80
MODERNBERT_MODEL_ID = "Lucius-Morningstar/mailroom-modernbert-classifier"


def gpu_usd_per_hour(gpu: str = "L4") -> float:
    raw = (os.environ.get("MODAL_GPU_USD_PER_HOUR") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    per_sec = (os.environ.get("MODAL_GPU_USD_PER_SEC") or "").strip()
    if per_sec:
        try:
            return float(per_sec) * 3600.0
        except ValueError:
            pass
    return L4_USD_PER_HOUR if gpu.split(":")[0] == "L4" else L4_USD_PER_HOUR


def estimate_gpu_cost_usd(gpu_seconds: float, *, gpu: str = "L4") -> float | None:
    if gpu_seconds <= 0:
        return None
    return round(gpu_seconds / 3600.0 * gpu_usd_per_hour(gpu), 6)


def attach_run_telemetry(
    report: dict[str, Any],
    *,
    remote_wall_seconds: float,
    wall_seconds_total: float | None = None,
    modal_profile: str | None = None,
    modal_app: str = "mailroom-ml-eval",
    gpu: str = "L4",
    modal_run_url: str | None = None,
) -> dict[str, Any]:
    """Merge ``run_telemetry`` + dojo-shaped ``serving`` fields into ``report``."""
    n = int(report.get("n_docs") or 0)
    gpu_cost = estimate_gpu_cost_usd(remote_wall_seconds, gpu=gpu)
    e2e = round(remote_wall_seconds / n, 6) if n else None
    gpu_cpd = round(gpu_cost / n, 8) if gpu_cost is not None and n else None
    telemetry = {
        "wall_seconds_total": round(wall_seconds_total or remote_wall_seconds, 3),
        "remote_wall_seconds": round(remote_wall_seconds, 3),
        "latency_seconds_per_document": e2e,
        "modal_profile": modal_profile,
        "modal_app": modal_app,
        "modal_run_url": modal_run_url,
        "gpu": gpu,
        "estimated_gpu_cost_usd": gpu_cost,
        "gpu_cost_per_document": gpu_cpd,
    }
    report["run_telemetry"] = telemetry
    return report


def experiment_record_from_report(
    report: Mapping[str, Any],
    *,
    experiment_name: str,
    modal_profile: str | None = None,
) -> dict[str, Any]:
    """Shape aligned with sandbox ``experiment_log.new_record`` + API run reports."""
    tel = report.get("run_telemetry") or {}
    n = int(report.get("n_docs") or 0)
    scores = {
        "accuracy": report.get("doc_type_accuracy"),
        "exact_match": report.get("doc_type_accuracy"),
        "doc_type_accuracy": report.get("doc_type_accuracy"),
        "subclass_accuracy": report.get("subclass_accuracy_conditional"),
    }
    subset = report.get("eval_subset") or "test"
    sample = report.get("sample")
    fingerprint = (
        f"modernbert-{subset}-n{n}-per_stratum{sample}-"
        f"seed{report.get('seed', 42)}-{report.get('artifact_sha', 'sha')[:12]}"
    )
    rec: dict[str, Any] = {
        "experiment_name": experiment_name,
        "task": "modernbert_eval",
        "suite": "modernbert_eval",
        "profile": modal_profile or tel.get("modal_profile") or "modal",
        "provider": "modal-gpu",
        "serving_kind": "modernbert",
        "classifier": "modernbert",
        "model": MODERNBERT_MODEL_ID,
        "mock": False,
        "dataset_fingerprint": fingerprint,
        "finetune_revision": report.get("finetune_revision"),
        "eval_subset": subset,
        "sample_per_stratum": sample,
        "eval_split_counts": report.get("eval_split_counts"),
        "n": n,
        "scores": scores,
        "wall_seconds": tel.get("remote_wall_seconds") or tel.get("wall_seconds_total"),
        "e2e_latency_seconds": tel.get("latency_seconds_per_document"),
        "estimated_gpu_cost_usd": tel.get("estimated_gpu_cost_usd"),
        "gpu_cost_per_document": tel.get("gpu_cost_per_document"),
        "gpu": tel.get("gpu"),
        "checkpoint": report.get("checkpoint"),
        "model_kind": report.get("model_kind"),
        "artifact_sha": report.get("artifact_sha"),
        "window_calibration": report.get("window_calibration"),
        "fast_path_rate": report.get("fast_path_rate"),
        "recorded_gates": report.get("recorded_gates"),
        "modal_app": tel.get("modal_app"),
        "modal_run_url": tel.get("modal_run_url"),
    }
    return {k: v for k, v in rec.items() if v is not None}
