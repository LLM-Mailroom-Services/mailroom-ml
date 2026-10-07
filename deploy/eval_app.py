"""Modal eval app for the mailroom-ml ModernBERT classifier — ``mailroom-ml-eval``.

Runs the documented eval CLI (``training/eval_modernbert.py``, plan §11,
issue #92 M7) on GPU where the trained checkpoint lives, so the selective-risk
threshold sweep runs at deployment context (8,192 tokens) in minutes instead
of CPU-hours.

    HF_TOKEN=... modal run deploy/eval_app.py --sample 50 --seed 42
    HF_TOKEN=... modal run deploy/eval_app.py --module runs/<run-id> --json
    HF_TOKEN=... modal run deploy/eval_app.py --module runs/<run-id> --sample 0 \\
        --as-json --decode-adjust-sweep 0.25,0.5,0.75,1.0 --out reports/json

- mounts the checkpoint Volume (``modernbert-checkpoints`` — the trainer's
  ``latest/`` pointer or any archived ``runs/<run-id>/``),
- pulls the pinned training/eval stage from the Hub at runtime (same pin as
  ``config.TRAINING_DATA_REVISION`` — never an empty local bake),
- invokes ``eval_modernbert.py --checkpoint <vol> --stage /root/stage
  --sample N --seed S [--selective-risk] [--json]`` inside an L4 container.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

_DEPLOY_DIR = Path(__file__).resolve().parent

APP_NAME = "mailroom-ml-eval"
CHECKPOINT_VOLUME_NAME = "modernbert-checkpoints"
HF_CACHE_VOLUME_NAME = "mailroom-ml-hf-cache"
CHECKPOINT_MOUNT = "/checkpoints"
HF_CACHE_MOUNT = "/root/.cache/huggingface"
STAGE_MOUNT = "/root/stage"
EVAL_SCRIPT = "/root/training/eval/eval_modernbert.py"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mailroom_ml.config import (  # noqa: E402
    TRAINING_DATA_REPO,
    TRAINING_DATA_REVISION,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTHONPATH": "/root/src", "HF_HOME": HF_CACHE_MOUNT})
    .uv_pip_install(
        "torch>=2.14",
        "transformers>=5.18",
        "pandas>=3.0.6",
        "numpy>=2.2",
        "pyarrow>=25.0",
        "onnxruntime>=1.30",
        "huggingface_hub>=1.33,<2",
        "datasets>=5.0.1",
    )
    .add_local_dir(ROOT / "src", remote_path="/root/src")
    .add_local_dir(ROOT / "training", remote_path="/root/training")
)

# heldout-plus v1 extension (local build artifact, gitignored): mount only
# when present so other subsets keep working on fresh checkouts.
_HELDOUT_PLUS_LOCAL = ROOT / "data" / "heldout_plus_v1"
if _HELDOUT_PLUS_LOCAL.is_dir():
    image = image.add_local_dir(
        _HELDOUT_PLUS_LOCAL, remote_path="/root/data/heldout_plus_v1"
    )

checkpoint_vol = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

_DEPLOY_ENV_KEYS = ("HF_TOKEN",)


def _config_secrets() -> list[modal.Secret]:
    values = {k: os.environ.get(k) for k in _DEPLOY_ENV_KEYS if os.environ.get(k)}
    if not values:
        return []
    return [modal.Secret.from_dict(values)]


app = modal.App(APP_NAME, image=image, tags={
    "project": "digital-mailroom",
    "package": "mailroom-ml",
    "purpose": "modernbert-eval",
})


def _telemetry_dir_for_eval_json(dest: Path) -> Path:
    """Sidecars live under ``reports/TEST-EVAL/telemetry/<eval-stem>/``."""
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "reports" / "TEST-EVAL" / "telemetry" / dest.stem


def _parse_eval_report_stdout(stdout: str) -> dict:
    """Extract the eval JSON report from mixed stdout (progress lines + JSON)."""
    text = stdout.strip()
    if not text:
        raise ValueError("eval_modernbert produced empty stdout")
    try:
        report = json.loads(text)
        if isinstance(report, dict) and "n_docs" in report:
            return report
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            report, _end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(report, dict) and "n_docs" in report:
            return report
    raise ValueError("eval_modernbert stdout contained no eval JSON report")


def _ensure_stage_tree() -> None:
    """Download the pinned training dataset to ``STAGE_MOUNT`` (train app parity)."""
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN absent — redeploy eval_app with HF_TOKEN exported so the "
            "Secret is captured (same as mailroom-ml-train).")
    from huggingface_hub import HfApi, snapshot_download  # noqa: PLC0415

    info = HfApi().dataset_info(TRAINING_DATA_REPO, revision=TRAINING_DATA_REVISION)
    if not str(getattr(info, "sha", "")).startswith(TRAINING_DATA_REVISION):
        raise RuntimeError(
            f"dataset pin mismatch: requested {TRAINING_DATA_REVISION}, "
            f"Hub reports {info.sha}")
    marker = Path(STAGE_MOUNT) / "documents" / "test"
    if not marker.is_dir() or not list(marker.glob("*.parquet")):
        print(f"[mailroom-ml-eval] pulling {TRAINING_DATA_REPO} "
              f"@ {TRAINING_DATA_REVISION} -> {STAGE_MOUNT}", flush=True)
        snapshot_download(
            repo_id=TRAINING_DATA_REPO,
            repo_type="dataset",
            revision=TRAINING_DATA_REVISION,
            local_dir=STAGE_MOUNT,
        )
    else:
        print(f"[mailroom-ml-eval] stage cache hit under {STAGE_MOUNT}", flush=True)


def _eval_cli_cmd(
        module: str, sample: int, seed: int, subset: str, *,
        selective_risk: bool, as_json: bool,
        subclass_decode_logit_adjust: float = 0.0,
        eval_extra: str = "") -> list[str]:
    """Build the eval_modernbert.py argv (d=0 is shipped argmax)."""
    cmd = [
        sys.executable,
        EVAL_SCRIPT,
        "--checkpoint", f"{CHECKPOINT_MOUNT}/{module}",
        "--stage", STAGE_MOUNT,
        "--subset", subset,
        "--sample", str(sample),
        "--seed", str(seed),
        "--subclass-decode-logit-adjust", str(subclass_decode_logit_adjust),
    ]
    if selective_risk:
        cmd.append("--selective-risk")
    if as_json:
        cmd.append("--json")
    extra = (eval_extra or "").split()
    if extra:
        cmd.extend(extra)
    return cmd


def _parse_decode_adjust_sweep(raw: str) -> list[str]:
    """Comma-separated d values, preserving the operator's spelling."""
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def _run_one_eval_cli(cmd: list[str]) -> dict:
    print("[mailroom-ml-eval] " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr.strip():
        print(result.stderr, end="", flush=True)
    if result.returncode != 0:
        if result.stdout.strip():
            print(result.stdout[-4000:], flush=True)
        print(result.stderr[-4000:], flush=True)
        raise RuntimeError(f"eval_modernbert.py exited {result.returncode}")
    out: dict = {"returncode": result.returncode}
    if "--json" in cmd and result.stdout.strip():
        out["report"] = _parse_eval_report_stdout(result.stdout)
    return out


@app.function(
    gpu="L4",
    volumes={
        CHECKPOINT_MOUNT: checkpoint_vol,
        HF_CACHE_MOUNT: hf_cache_vol,
    },
    timeout=60 * 120,
    startup_timeout=60 * 10,
    secrets=_config_secrets(),
)
def run_eval(module: str = "latest", sample: int = 50, seed: int = 42,
             subset: str = "test", selective_risk: bool = True,
             as_json: bool = False,
             subclass_decode_logit_adjust: float = 0.0,
             eval_extra: str = "",
             decode_adjust_sweep: str = "") -> dict:
    """Run the eval CLI against a checkpoint on the Volume.

    ``decode_adjust_sweep`` (e.g. ``0.25,0.5,0.75,1.0``) runs every d in
    one L4 container. Default ``subclass_decode_logit_adjust=0`` is current
    argmax; do not re-run d=0 unless it is listed in the sweep.
    """
    remote_started = time.perf_counter()
    if subset == "test":
        _ensure_stage_tree()
    sweep = _parse_decode_adjust_sweep(decode_adjust_sweep)
    if sweep:
        reports: dict[str, dict] = {}
        for d_str in sweep:
            cmd = _eval_cli_cmd(
                module, sample, seed, subset,
                selective_risk=selective_risk, as_json=as_json,
                subclass_decode_logit_adjust=float(d_str),
                eval_extra=eval_extra)
            one = _run_one_eval_cli(cmd)
            if as_json:
                reports[d_str] = one.get("report") or {}
        remote_wall = time.perf_counter() - remote_started
        return {
            "returncode": 0,
            "module": module,
            "sample": sample,
            "seed": seed,
            "subset": subset,
            "remote_wall_seconds": round(remote_wall, 3),
            "gpu": "L4",
            "decode_adjust_sweep": sweep,
            "reports": reports,
        }
    cmd = _eval_cli_cmd(
        module, sample, seed, subset,
        selective_risk=selective_risk, as_json=as_json,
        subclass_decode_logit_adjust=subclass_decode_logit_adjust,
        eval_extra=eval_extra)
    one = _run_one_eval_cli(cmd)
    remote_wall = time.perf_counter() - remote_started
    out: dict = {
        "returncode": one["returncode"],
        "module": module,
        "sample": sample,
        "seed": seed,
        "subset": subset,
        "remote_wall_seconds": round(remote_wall, 3),
        "gpu": "L4",
        "subclass_decode_logit_adjust": subclass_decode_logit_adjust,
    }
    if as_json and one.get("report"):
        out["report"] = one["report"]
    return out


def _active_modal_profile() -> str | None:
    try:
        return modal.config.get_profile()
    except Exception:
        return os.environ.get("MODAL_PROFILE")


def _local_helpers():
    if str(_DEPLOY_DIR) not in sys.path:
        sys.path.insert(0, str(_DEPLOY_DIR))
    from eval_telemetry import attach_run_telemetry, experiment_record_from_report
    from eval_tracing import (
        configure_tracing,
        eval_span,
        flush_tracing,
        per_doc_trace_rows,
    )

    return (
        attach_run_telemetry,
        experiment_record_from_report,
        configure_tracing,
        eval_span,
        flush_tracing,
        per_doc_trace_rows,
    )


def _write_eval_json(dest: Path, report: dict, *,
                     attach_run_telemetry, experiment_record_from_report,
                     configure_tracing, eval_span, flush_tracing,
                     per_doc_trace_rows, tracer, name: str, profile,
                     remote_wall: float, wall_total: float) -> None:
    attach_run_telemetry(
        report,
        remote_wall_seconds=float(remote_wall or wall_total),
        wall_seconds_total=wall_total,
        modal_profile=profile,
        modal_app=APP_NAME,
    )
    experiment_record = experiment_record_from_report(
        report, experiment_name=name, modal_profile=profile)
    experiment_record["tracing_backend"] = (
        "phoenix" if os.environ.get("MODERNBERT_TRACE_SINK", "auto") != "none" else "none"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"[mailroom-ml-eval] wrote {dest}", flush=True)
    telem = _telemetry_dir_for_eval_json(dest)
    telem.mkdir(parents=True, exist_ok=True)
    rec_path = telem / "experiment_record.json"
    rec_path.write_text(
        json.dumps(experiment_record, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[mailroom-ml-eval] wrote {rec_path}", flush=True)
    trace_path = telem / "otel_spans.jsonl"
    flush_tracing(tracer, out_path=trace_path)
    manifest_path = telem / "otel_spans.manifest.json"
    if trace_path.is_file():
        manifest_path.write_text(
            json.dumps(
                {
                    "span_file": str(trace_path.relative_to(telem)),
                    "n_lines": sum(1 for _ in trace_path.open()),
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    doc_rows = per_doc_trace_rows(report.get("per_doc") or [], run_id=name)
    doc_trace = telem / "per_doc_trace.jsonl"
    doc_trace.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in doc_rows) + "\n",
        encoding="utf-8",
    )
    print(f"[mailroom-ml-eval] wrote {doc_trace} ({len(doc_rows)} docs)", flush=True)
    report["telemetry_paths"] = {
        "experiment_record": str(rec_path),
        "otel_spans": str(trace_path),
        "per_doc_trace": str(doc_trace),
    }
    dest.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        f"[mailroom-ml-eval] wall={wall_total:.1f}s "
        f"gpu_est=${experiment_record.get('estimated_gpu_cost_usd')} "
        f"doc_type_acc={report.get('doc_type_accuracy')}",
        flush=True,
    )


@app.local_entrypoint()
def main(module: str = "latest", sample: int = 50, seed: int = 42,
         subset: str = "test", selective_risk: bool = True, as_json: bool = False,
         out: str = "", experiment_name: str = "",
         subclass_decode_logit_adjust: float = 0.0,
         eval_extra: str = "",
         decode_adjust_sweep: str = "") -> None:
    (
        attach_run_telemetry,
        experiment_record_from_report,
        configure_tracing,
        eval_span,
        flush_tracing,
        per_doc_trace_rows,
    ) = _local_helpers()
    print(f"mailroom-ml-eval: module={module} subset={subset} sample={sample} "
          f"seed={seed} selective_risk={selective_risk} "
          f"d={subclass_decode_logit_adjust} sweep={decode_adjust_sweep or '-'}")
    profile = _active_modal_profile()
    name = experiment_name or f"modernbert_{subset}_sample{sample}_seed{seed}"
    tracer = configure_tracing(
        experiment_name=name,
        modal_profile=profile,
        environment=os.environ.get("OBSERVABILITY_ENVIRONMENT", "pilot"),
    )
    wall_started = time.perf_counter()
    with eval_span(
        tracer,
        "modernbert.modal_eval",
        experiment=name,
        modal_profile=profile or "",
        subset=subset,
        sample=str(sample),
        seed=str(seed),
        module=module,
    ):
        meta = run_eval.remote(
            module=module,
            sample=sample,
            seed=seed,
            subset=subset,
            selective_risk=selective_risk,
            as_json=as_json,
            subclass_decode_logit_adjust=subclass_decode_logit_adjust,
            eval_extra=eval_extra,
            decode_adjust_sweep=decode_adjust_sweep,
        )
    wall_total = time.perf_counter() - wall_started
    run_id = Path(module).name
    helpers = dict(
        attach_run_telemetry=attach_run_telemetry,
        experiment_record_from_report=experiment_record_from_report,
        configure_tracing=configure_tracing,
        eval_span=eval_span,
        flush_tracing=flush_tracing,
        per_doc_trace_rows=per_doc_trace_rows,
        tracer=tracer,
        profile=profile,
        remote_wall=float(meta.get("remote_wall_seconds") or wall_total),
        wall_total=wall_total,
    )
    if as_json and meta.get("reports"):
        out_dir = Path(out) if out else Path("reports/json")
        if out_dir.suffix == ".json":
            out_dir = out_dir.parent
        for d_str, report in meta["reports"].items():
            dest = out_dir / f"eval_{run_id}-d{d_str}.json"
            d_name = experiment_name or (
                f"modernbert_{subset}_n{report.get('n_docs')}_d{d_str}")
            _write_eval_json(dest, report, name=d_name, **helpers)
        return
    if as_json and meta.get("report"):
        report = meta["report"]
        if out:
            dest = Path(out)
            name = experiment_name or (
                f"modernbert_{subset}_n{report.get('n_docs')}_seed{seed}")
            _write_eval_json(dest, report, name=name, **helpers)
        else:
            flush_tracing(tracer, out_path=None)
            print(json.dumps(report, sort_keys=True, indent=2))
            print(
                f"[mailroom-ml-eval] wall={wall_total:.1f}s "
                f"doc_type_acc={report.get('doc_type_accuracy')}",
                flush=True,
            )
