"""Modal training app for the mailroom-ml ModernBERT fine-tune — ``mailroom-ml-train``.

Port of the committed predecessor app (``Mailroom-Corpus-EDA @ cf096fa``,
``modernbert/modal_app.py``) updated for the mailroom-ml package.

Verified against the current Modal SDK on 2026-09-18:

- SDK version: **modal 1.5.5** (2026-08-28) — https://modal.com/docs/sdk/py/releases
- GPUs are configured **by string** (e.g. ``gpu="L4"``, ``gpu="H100:2"``) — the
  committed app's comment is confirmed correct; the object API
  (``modal.gpu.L4()``) no longer exists in the 1.x line. See
  https://modal.com/docs/examples/flux and /docs/examples/gpu_fallbacks.
- ``modal.App(name, tags=...)`` — App tags landed in 1.2.0 (release notes).
- ``modal.Image.debian_slim(python_version="3.12")``, ``.uv_pip_install()``,
  ``.add_local_dir(path, remote_path=)``, ``.env({...})`` — current reference API.
- ``modal.Volume.from_name(name, create_if_missing=True)`` + ``.commit()`` —
  current reference API (lazy hydration; no network at construction time).
- ``modal.Secret.from_dict({...})`` — current reference API.
- ``startup_timeout=`` on ``@app.function`` — added in 1.1.4 (release notes).

Behavior (mirrors the committed app):

- bundles ``src/`` + ``training/`` + ``configs/`` into the image at deploy time,
- pulls the published training dataset from ``config.TRAINING_DATA_REPO`` — never
  a local copy; the pinned ``config.TRAINING_DATA_REVISION`` is the contract and
  is validated (HfApi) plus exported as ``TRAINING_DATA_REVISION`` env before the
  trainer runs,
- invokes ``training/train_modernbert.py`` via subprocess with the exact
  documented CLI surface (--data, --output, --epochs, --batch-size, --grad-accum,
  --lr, --seed, --push-to-hub, --eval-test),
- persists the checkpoint to the ``modernbert-checkpoints`` Volume (stable
  ``latest/`` pointer + per-run archive under ``runs/<run-id>/`` for rollback) and
  pushes the checkpoint to the Hub when ``--push-to-hub`` is set,
- HF_TOKEN arrives via the secret (deploy-time env or the Modal dashboard).

Deploy (only HF_TOKEN is required — the training dtype is bf16-on-cuda):

    HF_TOKEN=... uv run --extra deploy modal deploy deploy/modal_app.py

Run (defaults: 5 epochs, batch 4, grad-accum 8, lr 2e-5, seed 42, eval-test on):

    HF_TOKEN=... uv run --extra deploy modal run deploy/modal_app.py
    HF_TOKEN=... uv run --extra deploy modal run deploy/modal_app.py \\
        --epochs 5 --push-to-hub Lucius-Morningstar/mailroom-modernbert-classifier
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal

# Package identity -----------------------------------------------------------
APP_NAME = "mailroom-ml-train"
MODAL_SDK_VERSION = "1.5.5"  # verified 2026-09-18 (docs releases page + PyPI)

# Named resources (Volume names are shared with the committed predecessor so an
# existing checkpoint corpus stays readable; the hf-cache is re-homed).
CHECKPOINT_VOLUME_NAME = "modernbert-checkpoints"
HF_CACHE_VOLUME_NAME = "mailroom-ml-hf-cache"
CHECKPOINT_MOUNT = "/checkpoints"
HF_CACHE_MOUNT = "/root/.cache/huggingface"
TRAINER_SCRIPT = "/root/training/train_modernbert.py"

# GPU + timeouts — exposed as constants so the deploy test suite can assert the
# deployed configuration offline (string GPU API, verified against 1.5.5 docs).
TRAIN_GPU = "L4"
TRAIN_TIMEOUT_S = 60 * 60 * 4   # 4h: weight download + 5 epochs on 5k windows
TRAIN_STARTUP_TIMEOUT_S = 60 * 10

# Deploy-time env keys copied into a Modal Secret (never hardcode tokens).
_DEPLOY_ENV_KEYS = ("HF_TOKEN",)

# Repo root — the deploy file lives at <root>/deploy/modal_app.py.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # repo-local bootstrap (mirrors committed train.py)

from mailroom_ml.config import (  # noqa: E402  (contract: single source of truth)
    MODEL_ID,
    TRAINING_DATA_REPO,
    TRAINING_DATA_REVISION,
)

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
# deps mirror the committed training stack + this package's train extra
# (pyproject.toml): torch/transformers for the trainer, datasets+huggingface_hub
# for the Hub data pull, scipy for temperature scaling.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .env(
        {
            "PYTHONPATH": "/root/src",
            "MAILROOM_ML_ROOT": "/root",  # training bootstrap anchor (cf096fa pattern)
        }
    )
    .uv_pip_install(
        "torch>=2.4",
        "transformers>=4.48",  # ModernBERT requires >= 4.48 (model card)
        "accelerate>=1.0",
        "datasets>=2.19",
        "evaluate>=0.4",
        "scipy>=1.11",
        "pandas>=2.2",
        "numpy>=1.26",
        "pyarrow>=15.0",
        "pyyaml>=6.0",
        "tqdm>=4.0",
        "huggingface_hub>=0.24",
    )
    .add_local_dir(ROOT / "src", remote_path="/root/src")
    .add_local_dir(ROOT / "training", remote_path="/root/training")
    .add_local_dir(ROOT / "configs", remote_path="/root/configs")
)

checkpoint_vol = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

app = modal.App(
    APP_NAME,
    image=image,
    tags={
        "project": "digital-mailroom",
        "package": "mailroom-ml",
        "purpose": "modernbert-train",
    },
)


def _config_secrets() -> list[modal.Secret]:
    """Secret from deploy-time env (committed pattern; verified current API)."""
    values = {k: os.environ.get(k) for k in _DEPLOY_ENV_KEYS if os.environ.get(k)}
    if not values:
        return []
    return [modal.Secret.from_dict(values)]


def _build_train_cmd(
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seed: int,
    push_to_hub: str,
    eval_test: bool,
) -> list[str]:
    """The exact trainer CLI invocation — the documented train_modernbert.py
    surface is held here, byte for flag, so the deploy test suite can assert it.

    Flags surfaced by the job: --data, --output, --epochs, --batch-size,
    --grad-accum, --lr, --seed, --seed, --push-to-hub <repo>, --eval-test.
    """
    cmd = [
        sys.executable,
        TRAINER_SCRIPT,
        "--data",
        TRAINING_DATA_REPO,
        "--output",
        f"{CHECKPOINT_MOUNT}/latest",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--grad-accum",
        str(grad_accum),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
    ]
    if push_to_hub:
        cmd += ["--push-to-hub", push_to_hub]
    if eval_test:
        cmd += ["--eval-test"]
    return cmd


@app.function(
    gpu=TRAIN_GPU,  # string API — verified current for the 1.5.5 SDK
    volumes={CHECKPOINT_MOUNT: checkpoint_vol, HF_CACHE_MOUNT: hf_cache_vol},
    secrets=_config_secrets(),
    timeout=TRAIN_TIMEOUT_S,
    startup_timeout=TRAIN_STARTUP_TIMEOUT_S,
)
def train(
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 2e-5,
    seed: int = 42,
    push_to_hub: str = "",
    eval_test: bool = True,
) -> dict:
    """Run the fine-tune inside an L4 GPU container.

    Seams:
    - data pin: TRAINING_DATA_REPO/TRAINING_DATA_REVISION come from
      src/mailroom_ml/config.py (bundled at /root/src). The revision is checked
      against the live Hub before the trainer starts so a broken pin fails
      before GPU minutes are spent, and is exported as TRAINING_DATA_REVISION
      for any training-layer revision support.
    - checkpoint: trainer writes to /checkpoints/latest; on success the run is
      archived to /checkpoints/runs/<run-id>/ (rollback), then the Volume is
      committed once.
    """
    from huggingface_hub import HfApi

    print(f"[mailroom-ml-train] config: data={TRAINING_DATA_REPO} "
          f"rev={TRAINING_DATA_REVISION} model={MODEL_ID}", flush=True)

    # Contract check — pin must resolve on the live Hub (fail fast, pre-GPU).
    info = HfApi().dataset_info(TRAINING_DATA_REPO, revision=TRAINING_DATA_REVISION)
    if not str(getattr(info, "sha", "")).startswith(TRAINING_DATA_REVISION):
        raise RuntimeError(
            f"dataset pin mismatch: requested {TRAINING_DATA_REVISION}, "
            f"Hub reports {info.sha} — do not train on an unpinned dataset"
        )
    print(f"[mailroom-ml-train] dataset pin verified: {info.sha}", flush=True)

    cmd = _build_train_cmd(epochs, batch_size, grad_accum, lr, seed,
                           push_to_hub, eval_test)
    print("[mailroom-ml-train] " + " ".join(cmd), flush=True)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"train_modernbert.py exited {result.returncode}")

    # Rollback: keep every successful run under runs/<run-id>/; latest/ stays
    # the stable pointer the export/serve steps consume.
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    archive_dir = f"{CHECKPOINT_MOUNT}/runs/{run_id}"
    shutil.copytree(f"{CHECKPOINT_MOUNT}/latest", archive_dir)
    checkpoint_vol.commit()
    print(f"[mailroom-ml-train] archived checkpoint to {archive_dir}", flush=True)

    return {
        "returncode": result.returncode,
        "run_id": run_id,
        "archive_dir": archive_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "seed": seed,
        "push_to_hub": push_to_hub,
        "eval_test": eval_test,
    }


@app.local_entrypoint()
def main(
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 2e-5,
    seed: int = 42,
    push_to_hub: str = "",
    eval_test: bool = True,
) -> None:
    print(f"mailroom-ml-train: epochs={epochs} batch={batch_size} "
          f"grad_accum={grad_accum} lr={lr} seed={seed} "
          f"push={push_to_hub or 'no'}")
    train.remote(epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
                 lr=lr, seed=seed, push_to_hub=push_to_hub, eval_test=eval_test)
