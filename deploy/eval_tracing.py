"""OTLP (Phoenix) + local JSONL span export for ModernBERT Modal eval."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_SPANS: list[dict[str, Any]] = []


def _sink_kind() -> str:
    raw = (os.environ.get("MODERNBERT_TRACE_SINK") or os.environ.get("OBSERVABILITY_PROVIDER") or "auto").lower()
    if raw in {"none", "off", "disabled"}:
        return "none"
    if raw in {"file", "local"}:
        return "file"
    if raw == "phoenix":
        return "phoenix"
    # auto: Phoenix when endpoint reachable is checked by caller; default both
    return "auto"


def _phoenix_endpoint() -> str:
    return (os.environ.get("PHOENIX_ENDPOINT") or "http://localhost:6006/v1/traces").rstrip("/")


def _phoenix_reachable() -> bool:
    try:
        import urllib.request

        base = _phoenix_endpoint().replace("/v1/traces", "")
        req = urllib.request.Request(f"{base}/healthz", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def configure_tracing(
    *,
    service_name: str = "mailroom-ml-eval",
    environment: str = "pilot",
    experiment_name: str | None = None,
    modal_profile: str | None = None,
) -> Any:
    """Return an OTEL tracer, or None. Always enables local JSONL capture."""
    global _SPANS
    _SPANS = []
    kind = _sink_kind()
    use_phoenix = kind == "phoenix" or (kind == "auto" and _phoenix_reachable())
    use_file = kind in {"file", "auto"} or not use_phoenix

    resource_attrs = {
        "service.name": service_name,
        "deployment.environment": environment,
    }
    if experiment_name:
        resource_attrs["experiment.name"] = experiment_name
    if modal_profile:
        resource_attrs["modal.profile"] = modal_profile

    tracer = None
    if use_phoenix:
        try:
            from opentelemetry import trace as oteltrace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            exporter = OTLPSpanExporter(endpoint=_phoenix_endpoint(), headers={})
            provider = TracerProvider(resource=Resource.create(resource_attrs))
            provider.add_span_processor(BatchSpanProcessor(exporter))
            oteltrace.set_tracer_provider(provider)
            tracer = oteltrace.get_tracer("modernbert-eval", "1.0.0")
            tracer.sandbox_provider = provider  # type: ignore[attr-defined]
            tracer._eval_use_file = use_file  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"[mailroom-ml-eval] Phoenix OTLP unavailable ({exc}); file trace only", flush=True)
            use_file = True

    if use_file and tracer is None:
        class _FileTracer:
            def start_as_current_span(self, name: str, **kwargs):
                return _FileSpan(name, resource_attrs)

        tracer = _FileTracer()
    return tracer


class _FileSpan:
    def __init__(self, name: str, resource: dict[str, str]):
        self.name = name
        self.resource = resource
        self.attrs: dict[str, str] = {}
        self.start = time.time()
        self.end: float | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = str(value)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end = time.time()
        _SPANS.append(
            {
                "name": self.name,
                "start_unix": self.start,
                "end_unix": self.end,
                "duration_ms": round((self.end - self.start) * 1000, 3),
                "resource": self.resource,
                "attributes": self.attrs,
            }
        )
        return False


def _tracer_resource(tracer: Any) -> dict[str, str]:
    attrs = getattr(getattr(tracer, "resource", None), "attributes", None)
    return {str(k): str(v) for k, v in dict(attrs or {}).items()}


@contextmanager
def eval_span(tracer: Any, name: str, **attrs: str) -> Iterator[Any]:
    """Open a span on ``tracer``.

    A Phoenix/OTLP tracer built with ``_eval_use_file`` (``file`` / ``auto``
    sinks) ALSO mirrors the span into the local JSONL archive: that flag used
    to be set and never read, so ``otel_spans.jsonl`` came out empty whenever
    Phoenix was reachable.
    """
    if tracer is None:
        yield None
        return
    mirror = (_FileSpan(name, _tracer_resource(tracer))
              if getattr(tracer, "_eval_use_file", False) is True else None)
    try:
        with tracer.start_as_current_span(name) as span:
            for k, v in attrs.items():
                if v is None:
                    continue
                if mirror is not None:
                    mirror.set_attribute(k, v)
                if span is not None:
                    try:
                        span.set_attribute(k, v)
                    except Exception:
                        pass
            yield span
    finally:
        if mirror is not None:
            mirror.__exit__()


def flush_tracing(tracer: Any, *, out_path: Path | None = None,
                  flush_provider: bool = True) -> Path | None:
    """Flush OTLP and/or write local JSONL span archive.

    ``flush_provider=False`` only (re)writes the archive: a caller writing
    several archives (the decode-adjust sweep) flushes + shuts the provider
    down ONCE, on the last call, instead of per archive.
    """
    if flush_provider and tracer is not None \
            and hasattr(tracer, "sandbox_provider"):
        provider = tracer.sandbox_provider
        try:
            provider.force_flush()
            provider.shutdown()
        except Exception:
            pass
    if not _SPANS and out_path is None:
        return None
    if out_path is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in _SPANS]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    manifest = {
        "span_count": len(_SPANS),
        "phoenix_endpoint": _phoenix_endpoint() if _phoenix_reachable() else None,
        "spans_jsonl": str(out_path),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[mailroom-ml-eval] trace archive {out_path} ({len(_SPANS)} spans)", flush=True)
    return out_path


def per_doc_trace_rows(per_doc: list[dict], *, run_id: str) -> list[dict]:
    """Compact post-hoc rows from eval ``per_doc`` (for local analysis)."""
    rows = []
    for i, row in enumerate(per_doc):
        rows.append(
            {
                "name": "modernbert.classify_document",
                "run_id": run_id,
                "index": i,
                "filename": row.get("filename"),
                "gt_doc_type": row.get("gt_doc_type"),
                "pred_doc_type": row.get("pred_doc_type"),
                "dt_correct": row.get("dt_correct"),
                "sc_correct": row.get("sc_correct"),
                "n_windows": row.get("n_windows"),
                "agreement": row.get("agreement"),
                "fast_path": row.get("fast_path"),
            }
        )
    return rows
