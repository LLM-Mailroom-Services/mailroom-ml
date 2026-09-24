"""Optional Modal serving app for the ONNX int8 export — ``mailroom-ml-serve``.

**FALLBACK infrastructure.** The PLAN's primary serving path is the local ONNX
CPU session under ``artifacts/onnx/...`` (~1e-6 $/doc, µs–ms, zero per-call
cost). This app exists only as an escape hatch — e.g. a pipeline node that
cannot host a local session, or a demo endpoint — and runs the **same ONNX
int8 artifact** on Modal CPU (no GPU, no torch in this image).

Verified against the current Modal SDK on 2026-09-18 (modal 1.5.5):

- ``@modal.fastapi_endpoint`` is the current "turn a function into a web
  endpoint" API. The historical ``@modal.web_endpoint`` name was renamed to
  ``@modal.fastapi_endpoint`` prior to v0.73.82 (Web Functions guide:
  https://modal.com/docs/guide/webhooks). The bearer-token pattern below is
  the one from that same guide (HTTPBearer + env check).
- ``modal.App(name, image=, tags=)``, ``modal.Secret.from_dict/from_name``,
  ``modal.Volume.from_name(create_if_missing=True)`` — current reference APIs.

Model resolution order at runtime (``SERVE_MODEL_DIR`` env wins, then the
deploy-time-bundled artifacts, then the ``mailroom-ml-onnx`` Volume):

    1. /artifacts/onnx/model   bundled into the image at deploy (add_local_dir)
    2. /model-vol              on the mailroom-ml-onnx Volume (`modal volume put`)

The ONNX graph contract (see deploy/onnx_export.py): inputs ``input_ids`` and
``attention_mask`` (int64, dynamic batch + sequence), one logits output per
head named ``logits_<head>``. The label maps ship inline in the artifact bundle
(``labels.json``) — no Hub lookup at serve time.

Deploy (requires a bearer token; create the secret once):

    modal secret create mailroom-ml-serve-token SERVE_API_TOKEN=<random>
    uv run --extra deploy --extra serve modal deploy deploy/serve_app.py

Exercise:

    curl -H "Authorization: Bearer $SERVE_API_TOKEN" \\
         -H "Content-Type: application/json" \\
         -d '{"texts": ["Exhibit A: Software License Agreement ..."]}' \\
         https://<workspace>--mailroom-ml-serve-predict.modal.run

GET /health is metadata-only; POST /predict requires the bearer token.
"""
from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path

import modal
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

APP_NAME = "mailroom-ml-serve"
MODAL_SDK_VERSION = "1.5.5"  # verified 2026-09-18 (docs releases page + PyPI)
ONNX_VOLUME_NAME = "mailroom-ml-onnx"
ONNX_VOLUME_MOUNT = "/model-vol"
BUNDLED_MODEL_PATH = "/artifacts/onnx/model"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mailroom_ml.config import (  # noqa: E402
    ABSTAIN_UNKNOWN_CLASS,
    ARTIFACTS_DIR,
    CLASSIFIER_MODEL_REPO,
)

# model_id passed at deploy: SERVE_MODEL_ID env, else the package default repo.
SERVE_MODEL_ID = os.environ.get("SERVE_MODEL_ID", CLASSIFIER_MODEL_REPO)
MAX_LENGTH = int(os.environ.get("SERVE_MAX_LENGTH", "8192"))  # ModernBERT native

# ---------------------------------------------------------------------------
# Image: CPU-only ONNX runtime. Deliberately NO torch/transformers — the int8
# graph is ~147 MiB and loads in ms; dragging the training stack in just to serve
# it would make this fallback as heavy as the thing it replaces. Tokenization
# uses the standalone `tokenizers` Rust wheel (no torch dependency).
# ---------------------------------------------------------------------------
_ONNX_IMAGE_DEPS = (
    "fastapi[standard]>=0.115",
    "pydantic>=2.7",
    "onnxruntime>=1.18",
    "tokenizers>=0.19",
    "huggingface_hub>=0.24",
)

_local_onnx = ARTIFACTS_DIR / "onnx" / "model"


def _build_image() -> modal.Image:
    img = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        *_ONNX_IMAGE_DEPS
    )
    if _local_onnx.is_dir():  # bundle whatever export exists at deploy time
        img = img.add_local_dir(_local_onnx, remote_path=BUNDLED_MODEL_PATH)
    return img


image = _build_image()
onnx_vol = modal.Volume.from_name(ONNX_VOLUME_NAME, create_if_missing=True)

app = modal.App(
    APP_NAME,
    image=image,
    volumes={ONNX_VOLUME_MOUNT: onnx_vol},
    tags={
        "project": "digital-mailroom",
        "package": "mailroom-ml",
        "purpose": "modernbert-serve-fallback",
    },
)


def _serve_secrets() -> list[modal.Secret]:
    """Bearer token: deploy-time env wins, named Modal secret otherwise.

    The named secret is REQUIRED for a bare `modal deploy` — deployment fails
    loudly rather than shipping an unauthenticated internal endpoint.
    """
    if os.environ.get("SERVE_API_TOKEN"):
        return [modal.Secret.from_dict({"SERVE_API_TOKEN": os.environ["SERVE_API_TOKEN"]})]
    return [modal.Secret.from_name("mailroom-ml-serve-token", required_keys=["SERVE_API_TOKEN"])]


def _resolve_model_dir() -> str:
    """Order: SERVE_MODEL_DIR env > bundled artifacts > Volume mount."""
    override = os.environ.get("SERVE_MODEL_DIR")
    if override:
        return override
    if os.path.isdir(BUNDLED_MODEL_PATH):
        return BUNDLED_MODEL_PATH
    if os.path.isdir(ONNX_VOLUME_MOUNT):
        return ONNX_VOLUME_MOUNT
    raise FileNotFoundError(
        "no ONNX artifact found: run deploy/onnx_export.py so artifacts/onnx/model "
        "exists before deploying, or populate the mailroom-ml-onnx Volume via "
        "`modal volume put`"
    )


# -- runtime state -----------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _session(model_dir: str) -> dict:
    """Load the ONNX session + label maps + tokenizer once per container."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    mdir = Path(model_dir)
    quant = mdir / "model_quantized.onnx"
    fp32 = mdir / "model.onnx"
    if quant.is_file():
        onnx_file, quantized = str(quant), True
    elif fp32.is_file():
        onnx_file, quantized = str(fp32), False
    else:
        raise FileNotFoundError(f"no model.onnx / model_quantized.onnx under {mdir}")

    sess = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    maps = json.loads((mdir / "labels.json").read_text())
    tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_LENGTH)

    # pad id: tokenizer_config.json pad_token -> vocab, else config.json.
    pad_id = None
    tc = mdir / "tokenizer_config.json"
    if tc.is_file():
        pad_token = json.loads(tc.read_text()).get("pad_token")
        if pad_token is not None:
            pad_id = tok.token_to_id(pad_token)
    if pad_id is None:
        cfg = mdir / "config.json"
        if cfg.is_file():
            pad_id = json.loads(cfg.read_text()).get("pad_token_id")
    if pad_id is None:
        raise RuntimeError("cannot resolve pad token id from the artifact bundle")

    return {
        "session": sess,
        "maps": maps,
        "tokenizer": tok,
        "pad_id": pad_id,
        "quantized": quantized,
        "output_names": [o.name for o in sess.get_outputs()],
        "model_dir": str(mdir),
    }


def _encode(texts: list[str], state: dict):
    """Encode + right-pad to MAX_LENGTH; returns (ids, attention_mask) int64."""
    import numpy as np

    encs = state["tokenizer"].encode_batch(texts)
    ids = np.full((len(encs), MAX_LENGTH), state["pad_id"], dtype=np.int64)
    mask = np.zeros((len(encs), MAX_LENGTH), dtype=np.int64)
    for i, enc in enumerate(encs):
        n = min(len(enc.ids), MAX_LENGTH)
        ids[i, :n] = enc.ids[:n]
        mask[i, :n] = 1
    return ids, mask


def _subclass_prediction(
    logits_by_head: dict,
    maps: dict,
    dt_label: str,
    window_index: int,
) -> dict | None:
    """Conditional subclass head — abstain when doc_type has no subclass head."""
    if dt_label == ABSTAIN_UNKNOWN_CLASS or dt_label not in maps:
        return None
    head_key = f"logits_{dt_label}"
    if head_key not in logits_by_head:
        return None
    return _head_prediction(logits_by_head[head_key][window_index], maps, dt_label)


def _head_prediction(logits, maps: dict, head: str) -> dict:
    """argmax + per-label logits for one head using the inlined label maps."""
    id2label = {int(k): v for k, v in maps[head]["id2label"].items()}
    top = int(logits.argmax())
    return {
        "label": id2label[top],
        "logits": {id2label[i]: float(logits[i]) for i in range(len(id2label))},
    }


# -- endpoints ---------------------------------------------------------------
_bearer = HTTPBearer(auto_error=False)


def _auth_check(
    cred: HTTPAuthorizationCredentials | None = Depends(_bearer),  # noqa: B008 - FastAPI dependency-injection idiom
) -> None:
    token = os.environ.get("SERVE_API_TOKEN", "")
    if not token:
        raise HTTPException(status_code=503,
                            detail="SERVE_API_TOKEN unset in container")
    if cred is None or cred.credentials != token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="invalid bearer token")


_auth_dep = Depends(_auth_check)  # module-level singleton (FastAPI/ruff B008)


class PredictRequest(BaseModel):
    """POST /predict body: raw input strings, exactly as the model layer
    formats them (title + "\\n\\n" + window, v1 input format)."""

    texts: list[str] = Field(min_length=1, max_length=64)


@app.function(secrets=_serve_secrets(), timeout=120, startup_timeout=60)
@modal.fastapi_endpoint(method="POST")
def predict(request: PredictRequest, _: None = _auth_dep) -> dict:
    """Run the int8 ONNX session; returns per-text head predictions + logits."""
    state = _session(_resolve_model_dir())
    ids, mask = _encode(request.texts, state)
    outs = state["session"].run(None, {"input_ids": ids, "attention_mask": mask})
    logits_by_head = dict(zip(state["output_names"], outs, strict=True))
    head_names = sorted(k for k in logits_by_head if k.startswith("logits_"))

    predictions = []
    for i in range(len(request.texts)):
        dt = _head_prediction(logits_by_head["logits_doc_type"][i],
                              state["maps"], "doc_type")
        sc = _subclass_prediction(logits_by_head, state["maps"], dt["label"], i)
        predictions.append({"doc_type": dt, "subclass": sc})

    return {
        "model_id": SERVE_MODEL_ID,
        "artifact": state["model_dir"],
        "quantized": state["quantized"],
        "providers": state["session"].get_providers(),
        "heads": head_names,
        "predictions": predictions,
    }


@app.function(secrets=_serve_secrets(), timeout=60)
@modal.fastapi_endpoint()
def health() -> dict:
    """Readiness + artifact metadata (unauthenticated metadata only)."""
    try:
        model_dir = _resolve_model_dir()
        state = _session(model_dir)
        return {
            "status": "ok",
            "model_id": SERVE_MODEL_ID,
            "artifact": model_dir,
            "quantized": state["quantized"],
            "outputs": state["output_names"],
            "heads": sorted(state["maps"]),
        }
    except Exception as exc:  # noqa: BLE001 - health must never 500
        return {"status": "error", "detail": str(exc)}
