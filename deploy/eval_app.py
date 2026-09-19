"""Modal eval app for the mailroom-ml ModernBERT classifier — ``mailroom-ml-eval``.

Runs the documented eval CLI (``training/eval_modernbert.py``, plan §11,
issue #92 M7) on GPU where the trained checkpoint lives, so the selective-risk
threshold sweep runs at deployment context (8,192 tokens) in minutes instead
of CPU-hours.

    HF_TOKEN=... modal run deploy/eval_app.py --sample 50 --seed 42
    HF_TOKEN=... modal run deploy/eval_app.py --module runs/<run-id> --json

- mounts the checkpoint Volume (``modernbert-checkpoints`` — the trainer's
  ``latest/`` pointer or any archived ``runs/<run-id>/``),
- mounts the staged eval tree (``data/modernbert_training/stage`` — the
  pinned training repo's documents/test parquet; the held-out split never
  touches training/calibration),
- invokes ``eval_modernbert.py --checkpoint <vol> --stage /root/stage
  --sample N --seed S [--selective-risk] [--json]`` inside an L4 container.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "mailroom-ml-eval"
CHECKPOINT_VOLUME_NAME = "modernbert-checkpoints"
CHECKPOINT_MOUNT = "/checkpoints"
STAGE_MOUNT = "/root/stage"
EVAL_SCRIPT = "/root/training/eval_modernbert.py"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mailroom_ml.config import STAGE_DIR  # noqa: E402

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTHONPATH": "/root/src"})
    .uv_pip_install(
        "torch>=2.4",
        "transformers>=4.48",  # windower + pytorch bundle loading
        "pandas>=2.2",
        "numpy>=1.26",
        "pyarrow>=15.0",
        "onnxruntime>=1.17",  # ONNX bundles (prefer_onnx path)
    )
    .add_local_dir(ROOT / "src", remote_path="/root/src")
    .add_local_dir(ROOT / "training", remote_path="/root/training")
    .add_local_dir(STAGE_DIR, remote_path=STAGE_MOUNT)
)

checkpoint_vol = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME, image=image, tags={
    "project": "digital-mailroom",
    "package": "mailroom-ml",
    "purpose": "modernbert-eval",
})


@app.function(
    gpu="L4",
    volumes={CHECKPOINT_MOUNT: checkpoint_vol},
    timeout=60 * 60,
    startup_timeout=60 * 10,
)
def run_eval(module: str = "latest", sample: int = 50, seed: int = 42,
             selective_risk: bool = True, as_json: bool = False) -> dict:
    """Run the eval CLI against a checkpoint on the Volume."""
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
