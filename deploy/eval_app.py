"""Modal eval app for the mailroom-ml ModernBERT classifier — ``mailroom-ml-eval``.

Runs the documented eval CLI (``training/eval_modernbert.py``, plan §11,
issue #92 M7) on GPU where the trained checkpoint lives, so the selective-risk
threshold sweep runs at deployment context (8,192 tokens) in minutes instead
of CPU-hours.

    HF_TOKEN=... modal run deploy/eval_app.py --sample 50 --seed 42
    HF_TOKEN=... modal run deploy/eval_app.py --module runs/<run-id> --json

- mounts the checkpoint Volume (``modernbert-checkpoints`` — the trainer's
  ``latest/`` pointer or any archived ``runs/<run-id>/``),
- pulls the pinned training/eval stage from the Hub at runtime (same pin as
  ``config.TRAINING_DATA_REVISION`` — never an empty local bake),
- invokes ``eval_modernbert.py --checkpoint <vol> --stage /root/stage
  --sample N --seed S [--selective-risk] [--json]`` inside an L4 container.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "mailroom-ml-eval"
CHECKPOINT_VOLUME_NAME = "modernbert-checkpoints"
HF_CACHE_VOLUME_NAME = "mailroom-ml-hf-cache"
CHECKPOINT_MOUNT = "/checkpoints"
HF_CACHE_MOUNT = "/root/.cache/huggingface"
STAGE_MOUNT = "/root/stage"
EVAL_SCRIPT = "/root/training/eval_modernbert.py"

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
        "torch>=2.4",
        "transformers>=4.48",
        "pandas>=2.2",
        "numpy>=1.26",
        "pyarrow>=15.0",
        "onnxruntime>=1.17",
        "huggingface_hub>=0.24",
        "datasets>=2.19",
    )
    .add_local_dir(ROOT / "src", remote_path="/root/src")
    .add_local_dir(ROOT / "training", remote_path="/root/training")
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


@app.function(
    gpu="L4",
    volumes={
        CHECKPOINT_MOUNT: checkpoint_vol,
        HF_CACHE_MOUNT: hf_cache_vol,
    },
    timeout=60 * 60,
    startup_timeout=60 * 10,
    secrets=_config_secrets(),
)
def run_eval(module: str = "latest", sample: int = 50, seed: int = 42,
             selective_risk: bool = True, as_json: bool = False) -> dict:
    """Run the eval CLI against a checkpoint on the Volume."""
    _ensure_stage_tree()
    cmd = [
        sys.executable,
        EVAL_SCRIPT,
        "--checkpoint", f"{CHECKPOINT_MOUNT}/{module}",
        "--stage", STAGE_MOUNT,
        "--sample", str(sample),
        "--seed", str(seed),
    ]
    if selective_risk:
        cmd.append("--selective-risk")
    if as_json:
        cmd.append("--json")
    print("[mailroom-ml-eval] " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr[-4000:])
        raise RuntimeError(f"eval_modernbert.py exited {result.returncode}")
    return {"returncode": result.returncode, "module": module,
            "sample": sample, "seed": seed}


@app.local_entrypoint()
def main(module: str = "latest", sample: int = 50, seed: int = 42,
         selective_risk: bool = True, as_json: bool = False) -> None:
    print(f"mailroom-ml-eval: module={module} sample={sample} seed={seed} "
          f"selective_risk={selective_risk}")
    run_eval.remote(module=module, sample=sample, seed=seed,
                    selective_risk=selective_risk, as_json=as_json)
