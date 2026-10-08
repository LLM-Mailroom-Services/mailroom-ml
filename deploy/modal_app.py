"""Modal training app for the mailroom-ml ModernBERT fine-tune — ``mailroom-ml-train``.

Port of the committed predecessor app (``Mailroom-Corpus-EDA @ cf096fa``,
``modernbert/modal_app.py``) updated for the mailroom-ml package.

Verified against the current Modal SDK on 2026-10-02:

- SDK version: **modal 1.6.0** (2026-09-28) — https://modal.com/docs/sdk/py/releases
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

- bundles ``src/`` + ``training/`` + ``configs/`` + ``deploy/`` into the image
  at deploy time (``deploy/`` is required for the in-container ONNX export),
- pulls the published training dataset from ``config.TRAINING_DATA_REPO`` — never
  a local copy; the pinned ``config.TRAINING_DATA_REVISION`` is the contract and
  is validated (HfApi) plus exported as ``TRAINING_DATA_REVISION`` env before the
  trainer runs,
- invokes ``training/train/train_modernbert.py`` via subprocess with the exact
  documented CLI surface (--data, --output, --epochs, --batch-size, --grad-accum,
  --lr, --seed, --eval-test).  It does NOT pass ``--push-to-hub``: a trainer-side
  push would publish before ONNX export/parity and even from an ungated run,
- persists the checkpoint to the ``modernbert-checkpoints`` Volume (stable
  ``latest/`` pointer + per-run archive under ``runs/<run-id>/`` for rollback) and
  pushes the checkpoint to the Hub (``push_to_hub`` repo) only AFTER a
  successful ONNX export + parity AND only when the trainer's selection gate was
  met (``summary.json`` ``checkpoint_selection.gate_met``); smoke runs and
  resumes that trained nothing never promote or push,
- HF_TOKEN arrives via the secret (deploy-time env or the Modal dashboard).

Deploy (only HF_TOKEN is required — the training dtype is bf16-on-cuda):

    HF_TOKEN=... uv run --extra deploy modal deploy deploy/modal_app.py

Run (defaults: 5 epochs, batch 4, grad-accum 8, lr 2e-5, seed 42, eval-test on):

    HF_TOKEN=... uv run --extra deploy modal run deploy/modal_app.py
    HF_TOKEN=... uv run --extra deploy modal run deploy/modal_app.py \\
        --epochs 5 --push-to-hub Lucius-Morningstar/mailroom-modernbert-classifier

Long runs (1-3h): deploy once, then fire-and-forget via the spawn launcher —
a spawn on the deployed function is fully server-side and survives the
launching client dying (``modal run --detach`` does not):

    HF_TOKEN=... uv run --extra deploy modal deploy deploy/modal_app.py
    HF_TOKEN=... uv run --extra deploy python deploy/spawn_train.py \\
        --push-to-hub Lucius-Morningstar/mailroom-modernbert-classifier
    modal app logs ap-<app-id>   # epoch lines stream live
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal

# Package identity -----------------------------------------------------------
APP_NAME = "mailroom-ml-train"
MODAL_SDK_VERSION = "1.6.0"  # verified 2026-10-02 (docs releases page + PyPI)

# Named resources (Volume names are shared with the committed predecessor so an
# existing checkpoint corpus stays readable; the hf-cache is re-homed).
CHECKPOINT_VOLUME_NAME = "modernbert-checkpoints"
HF_CACHE_VOLUME_NAME = "mailroom-ml-hf-cache"
CHECKPOINT_MOUNT = "/checkpoints"
HF_CACHE_MOUNT = "/root/.cache/huggingface"
TRAINER_SCRIPT = "/root/training/train/train_modernbert.py"
# Hub upload hygiene (mirrors the trainer's own HUB_IGNORE_PATTERNS; 2026-09-20
# audit R10: optimizer/scheduler/resume state doubled the repo size).  ``onnx``
# is the serving bundle exported beside the run checkpoint: the model repo has
# only ever carried the PyTorch checkpoint (the ONNX bundle lives under
# ``artifacts/onnx/model`` / the checkpoint Volume), so it is not uploaded.
HUB_IGNORE_PATTERNS = ["optimizer.pt", "scheduler.pt", "resume.json",
                       "onnx", "onnx/*"]

# GPU + timeouts — exposed as constants so the deploy test suite can assert the
# deployed configuration offline (string GPU API, verified against 1.6.0 docs).
TRAIN_GPU = "L4"
# 8h: 5 epochs at dynamic padding measures ~1h/epoch on an L4 (~5h) +
# held-out test eval. A timeout here kills the run WITHOUT a checkpoint
# (saved only at the end), so the ceiling must clear the worst case.
TRAIN_TIMEOUT_S = 60 * 60 * 8
TRAIN_STARTUP_TIMEOUT_S = 60 * 10
# Compute guardrails (verified against the installed modal 1.6.0 SDK + docs on
# 2026-09-19): the default GPU-function request is 0.125 cores / 128 MiB — the
# first deployed run starved on 1 vCPU (GPU idle, epoch 1 never landed in 45
# min). cpu=8 physical cores (soft limit bursts to +16) and 16 GiB RAM keep the
# tokenizer/data-loading/optimizer work off the critical path; retries=0 means
# a half-trained run is never auto-rerun (a fresh spawn is the only retry).
#
# 2026-09-19 (cost pass): Modal bills CPU/memory on the REQUESTED allocation,
# and the trainer's per-step CPU work is ~0.5 s vs ~6 s of GPU work — 8 cores
# was ~3x over-provisioned. cpu=2 / 10 GiB keeps the GPU fed at ~25% lower
# cost/hour; the per-step log heartbeat makes any starvation visible within
# minutes (and per-epoch checkpoints cap the loss).
TRAIN_CPU = 2
TRAIN_MEMORY_MIB = 10240  # 10 GiB (container measured ~7.1 GiB peak)
TRAIN_RETRIES = 0

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
        "torch>=2.14",
        "transformers>=5.18",  # ModernBERT; floor tracks pyproject train extra
        "accelerate>=1.15",
        "datasets>=5.0.1",
        "evaluate>=0.4.6",
        "scipy>=1.17",
        "pandas>=3.0.6",
        "numpy>=2.2",
        "pyarrow>=25.0",
        "pyyaml>=6.0.3",
        "tqdm>=4.66",
        "huggingface_hub>=1.33,<2",  # datasets 5.0.1 caps the hub client at <2
        "onnxruntime>=1.30",
        "onnx>=1.19",
        "onnxscript>=0.7",
    )
    .add_local_dir(ROOT / "src", remote_path="/root/src")
    .add_local_dir(ROOT / "training", remote_path="/root/training")
    .add_local_dir(ROOT / "configs", remote_path="/root/configs")
    .add_local_dir(ROOT / "deploy", remote_path="/root/deploy")
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
    max_steps: int = 0,
    log_every: int = 0,
    resume: str = "",
    output: str = f"{CHECKPOINT_MOUNT}/latest",
    trainer_extra: list[str] | None = None,
) -> list[str]:
    """The exact trainer CLI invocation — the documented train_modernbert.py
    surface is held here, byte for flag, so the deploy test suite can assert it.

    Flags surfaced by the job: --data, --output, --epochs, --batch-size,
    --grad-accum, --lr, --seed, --push-to-hub <repo>, --eval-test,
    --max-steps (pre-flight smoke cap), --log-every (smoke step cadence —
    default 0 omits the flag and the trainer's own default of 50 applies),
    --resume <bundle dir> (continue a cut run from its last checkpoint —
    ``/checkpoints/runs/<run-id>``; ``latest/`` holds only the last
    SUCCESSFUL run).  ``push_to_hub`` is part of the documented trainer
    surface, but ``train()`` below always passes ``""`` and pushes itself
    after export + parity.
    ``trainer_extra`` appends arbitrary trainer flags verbatim (the
    2026-09-20 audit levers: --loss-lambda-dt, --label-smoothing,
    --weight-mode, --mlp-heads, --freeze-backbone-epochs, ...).
    """
    cmd = [
        sys.executable,
        TRAINER_SCRIPT,
        "--data",
        TRAINING_DATA_REPO,
        "--output",
        output,
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
    if max_steps:
        cmd += ["--max-steps", str(max_steps)]
    if log_every:
        cmd += ["--log-every", str(log_every)]
    if resume:
        cmd += ["--resume", resume]
    if trainer_extra:
        cmd += list(trainer_extra)
    return cmd


@app.function(
    gpu=TRAIN_GPU,  # string API — verified current for the 1.6.0 SDK
    cpu=TRAIN_CPU,  # 8 physical cores — 1-vCPU default starved the GPU (see above)
    memory=TRAIN_MEMORY_MIB,  # 16 GiB
    volumes={CHECKPOINT_MOUNT: checkpoint_vol, HF_CACHE_MOUNT: hf_cache_vol},
    secrets=_config_secrets(),
    timeout=TRAIN_TIMEOUT_S,
    startup_timeout=TRAIN_STARTUP_TIMEOUT_S,
    retries=TRAIN_RETRIES,  # never auto-rerun a half-trained run
)
def train(
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 2e-5,
    seed: int = 42,
    push_to_hub: str = "",
    eval_test: bool = True,
    max_steps: int = 0,
    log_every: int = 0,
    resume: str = "",
    trainer_extra: list[str] | None = None,
    export_onnx: bool = True,
) -> dict:
    """Run the fine-tune inside an L4 GPU container.

    Seams:
    - data pin: TRAINING_DATA_REPO/TRAINING_DATA_REVISION come from
      src/mailroom_ml/config.py (bundled at /root/src). The revision is checked
      against the live Hub before the trainer starts so a broken pin fails
      before GPU minutes are spent, and is exported as TRAINING_DATA_REVISION
      for any training-layer revision support.
    - checkpoint: the trainer writes to /checkpoints/runs/<run-id>/ (rollback
      archive); on success (+ ONNX parity) it is promoted to
      /checkpoints/latest, then the Volume is committed once.
    - hub push: NOT done by the trainer.  After export + parity (and promote),
      ``push_to_hub`` uploads the run iff the selection gate was met; an
      ungated run (selected_epoch == 0) or a resume that trained nothing is
      never pushed or promoted.
    - resume: pass a bundle dir (a cut run lives in /checkpoints/runs/<run-id>;
      latest/ only holds the last SUCCESSFUL run) to continue it from its last
      checkpoint instead of starting over.  A resume already at/after
      ``epochs`` trains nothing: no export, promote or push.
    - smoke (max_steps > 0): writes to /checkpoints/smoke-<ts> so the cadence
      probe never clobbers the real latest/ pointer.
    """
    from huggingface_hub import HfApi

    # R8 (2026-09-20 audit): the Secret is captured at DEPLOY time from the
    # shell env.  If HF_TOKEN was not exported then, the deployed function has
    # no token and every spawn dies at the dataset pull — fail loudly here
    # instead of silently.
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN absent in the container — the app was deployed without "
            "it exported, so the Secret was never captured. Re-deploy with "
            "`export HF_TOKEN=$(cat ~/.config/opencode/secrets/hf-token)`.")

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

    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    is_smoke = bool(max_steps)
    # Non-smoke trains into runs/<run-id> first.  latest/ is promoted only
    # after a successful train (+ optional ONNX parity).  A failed parity
    # check therefore leaves latest/ on the last good checkpoint (#19).
    output = (
        f"{CHECKPOINT_MOUNT}/smoke-{run_id}" if is_smoke
        else f"{CHECKPOINT_MOUNT}/runs/{run_id}"
    )
    # push_to_hub is deliberately NOT forwarded: the trainer would upload
    # before export/parity and even when no epoch cleared the selection gate.
    cmd = _build_train_cmd(epochs, batch_size, grad_accum, lr, seed,
                           "", eval_test, max_steps, log_every,
                           resume, output=output,
                           trainer_extra=trainer_extra)
    print("[mailroom-ml-train] " + " ".join(cmd), flush=True)

    # per-epoch checkpointing: the trainer commits the volume itself after
    # each epoch save (a kill/timeout before this app-level commit would
    # otherwise lose every epoch — 2026-09-19 incident).
    os.environ["MAILROOM_ML_CHECKPOINT_VOLUME"] = CHECKPOINT_VOLUME_NAME

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"train_modernbert.py exited {result.returncode}")

    archive_dir = ""
    onnx_dir = ""
    pushed = False
    promoted = False
    skip_reason = ""
    if is_smoke:
        skip_reason = "smoke run"
        print("[mailroom-ml-train] smoke run — skipping the runs/ archive "
              "and ONNX export (latest/ untouched)", flush=True)
    else:
        summary = _read_summary(output)
        if summary.get("nothing_trained"):
            skip_reason = "nothing trained"
            print("[mailroom-ml-train] resume was already at/after --epochs: "
                  "nothing trained — skipping ONNX export, promotion and "
                  "Hub push (latest/ untouched)", flush=True)
        else:
            archive_dir = output
            if export_onnx:
                onnx_dir = _export_onnx_and_parity(output)
            _promote_latest(output)
            promoted = True
            checkpoint_vol.commit()
            print(f"[mailroom-ml-train] archived checkpoint to {archive_dir}; "
                  f"latest/ promoted", flush=True)
            do_push, why = _push_decision(summary, push_to_hub,
                                          parity_ok=bool(onnx_dir))
            if do_push:
                _push_run_to_hub(output, push_to_hub, summary, epochs=epochs)
                pushed = True
            else:
                skip_reason = why
                if push_to_hub:
                    print(f"[mailroom-ml-train] NOT pushing to the Hub: {why}",
                          flush=True)

    return {
        "returncode": result.returncode,
        "run_id": run_id,
        "archive_dir": archive_dir,
        "onnx_dir": onnx_dir,
        "export_onnx": bool(onnx_dir),
        "promoted": promoted,
        "pushed": pushed,
        "skip_reason": skip_reason,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "seed": seed,
        "push_to_hub": push_to_hub,
        "eval_test": eval_test,
    }


def _read_summary(run_dir: str) -> dict:
    """The trainer's ``summary.json`` for a finished run (fail loud if absent:
    promote/push decisions are read from it)."""
    path = Path(run_dir) / "summary.json"
    if not path.is_file():
        raise RuntimeError(
            f"{path} missing after a successful trainer exit — cannot decide "
            "whether to promote/push; latest/ not updated")
    return json.loads(path.read_text(encoding="utf-8"))


def _push_decision(summary: dict, push_to_hub: str, *,
                   parity_ok: bool = True) -> tuple[bool, str]:
    """Whether to upload after a successful export + parity, and why not.

    Only a run whose selection gate was met (``checkpoint_selection.gate_met``,
    i.e. ``selected_epoch > 0``) is published: an ungated run ships the
    UNCALIBRATED final epoch.  A resume that trained nothing is never pushed,
    and neither is a run whose ONNX export + parity was skipped
    (``--no-export-onnx``): publishing is gated on parity.
    """
    if not push_to_hub:
        return False, "no --push-to-hub repo requested"
    if summary.get("nothing_trained"):
        return False, "nothing trained"
    if not parity_ok:
        return False, ("ONNX export + parity skipped (--no-export-onnx): "
                       "the Hub push is gated on parity")
    gate_met = bool(
        (summary.get("checkpoint_selection") or {}).get("gate_met", False))
    if not gate_met:
        return False, ("selection gate not met (selected_epoch == 0: "
                       "uncalibrated final-epoch weights)")
    return True, ""


def _push_run_to_hub(run_dir: str, repo: str, summary: dict, *,
                     epochs: int) -> None:
    """Upload the promoted run to ``repo`` (call only after export + parity)."""
    from huggingface_hub import HfApi

    selected = (summary.get("checkpoint_selection") or {}).get("epoch")
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=run_dir, repo_id=repo, repo_type="model",
        ignore_patterns=HUB_IGNORE_PATTERNS,
        commit_message=f"ModernBERT hierarchical classifier "
                       f"(epochs {epochs}, selected epoch {selected})")
    print(f"[mailroom-ml-train] pushed: https://huggingface.co/{repo}",
          flush=True)


def _export_onnx_and_parity(checkpoint_dir: str, *,
                            require_int8: bool = True) -> str:
    """Run the committed torch.onnx.export path + parity gate (#19).

    Never uses ``optimum-cli`` (random heads on this architecture).
    A non-zero exit leaves ``latest/`` untouched (caller has not promoted).
    """
    onnx_dir = str(Path(checkpoint_dir) / "onnx")
    export_cmd = [
        sys.executable, "/root/deploy/onnx_export.py",
        "--pytorch-dir", checkpoint_dir,
        "--out-dir", onnx_dir,
    ]
    print("[mailroom-ml-train] " + " ".join(export_cmd), flush=True)
    exported = subprocess.run(export_cmd)
    if exported.returncode != 0:
        raise RuntimeError(
            f"onnx_export.py exited {exported.returncode} — latest/ not updated")
    parity_cmd = [
        sys.executable, "/root/deploy/onnx_parity_check.py",
        "--pytorch-dir", checkpoint_dir,
        "--onnx-dir", onnx_dir,
    ]
    if require_int8:
        parity_cmd.append("--require-int8")
    print("[mailroom-ml-train] " + " ".join(parity_cmd), flush=True)
    parity = subprocess.run(parity_cmd)
    if parity.returncode != 0:
        raise RuntimeError(
            f"onnx_parity_check.py exited {parity.returncode} — "
            "latest/ not updated (fail loud)")
    return onnx_dir


def _promote_latest(run_dir: str, dest: str | None = None) -> None:
    """Replace ``latest/`` with a successful run via copy + rename (#19).

    Copy to a sibling ``*.promoting`` directory first, then swap.  A crash
    mid-copy leaves the previous ``latest/`` intact.  ``dest`` is injectable
    for tests; production uses ``{CHECKPOINT_MOUNT}/latest``.
    """
    latest = dest or f"{CHECKPOINT_MOUNT}/latest"
    tmp = f"{latest}.promoting"
    if Path(tmp).exists():
        shutil.rmtree(tmp)
    shutil.copytree(run_dir, tmp)
    backup = f"{latest}.bak"
    if Path(latest).exists():
        if Path(backup).exists():
            shutil.rmtree(backup)
        os.rename(latest, backup)
    try:
        os.rename(tmp, latest)
    except OSError:
        if Path(backup).exists() and not Path(latest).exists():
            os.rename(backup, latest)
        raise
    if Path(backup).exists():
        shutil.rmtree(backup)


@app.local_entrypoint()
def main(
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 2e-5,
    seed: int = 42,
    push_to_hub: str = "",
    eval_test: bool = True,
    export_onnx: bool = True,
) -> None:
    print(f"mailroom-ml-train: epochs={epochs} batch={batch_size} "
          f"grad_accum={grad_accum} lr={lr} seed={seed} "
          f"push={push_to_hub or 'no'} export_onnx={export_onnx}")
    train.remote(epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
                 lr=lr, seed=seed, push_to_hub=push_to_hub, eval_test=eval_test,
                 export_onnx=export_onnx)
