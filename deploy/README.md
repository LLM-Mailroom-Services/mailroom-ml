# mailroom-ml — Deployment Runbook

Deploy layer for the ModernBERT ingest fast-path classifier. Three pieces:

| File | App | What it does |
| --- | --- | --- |
| `modal_app.py` | `mailroom-ml-train` | One-shot L4 GPU training run (port of `Mailroom-Corpus-EDA @ cf096fa` `modernbert/modal_app.py`) |
| `serve_app.py` | `mailroom-ml-serve` | **FALLBACK** ONNX int8 CPU endpoint — the PLAN's primary serving path is the local ONNX session under `artifacts/onnx/…`; do not rely on this for production |
| `onnx_export.py` / `onnx_parity_check.py` | — | Checkpoint → ONNX (dynamic axes + int8) and the PyTorch-vs-ONNX logits parity gate |

## 1. Verified versions (checked 2026-09-18, before writing)

| Component | Version | Evidence |
| --- | --- | --- |
| Modal SDK | **1.5.5** (2026-08-28) | <https://modal.com/docs/sdk/py/releases> (live, fetched 2026-09-18) + PyPI json (`pypi.org/pypi/modal/json` → 1.5.5, `requires_python <3.15,>=3.10`) |
| GPU config | string API, e.g. `gpu="L4"` | current docs examples (flux / gpu_fallbacks / llm_inference) — the committed app's "Modal ≥1.0 configures GPUs by string" comment is **confirmed correct**; `modal.gpu.L4()` objects are gone |
| Web endpoints | `@modal.fastapi_endpoint` | current Web Functions guide — `@modal.web_endpoint` was renamed to `fastapi_endpoint` prior to v0.73.82 |
| `App(name, tags=)`, `Image.debian_slim(python_version=)`, `uv_pip_install`, `add_local_dir`, `.env()`, `Secret.from_dict`, `Volume.from_name(create_if_missing=True)` + `commit()`, `startup_timeout=` | current | `modal.App`/`modal.Image`/`modal.Secret`/`modal.Volume` reference pages (live) + 1.2.0 (tags) and 1.1.4 (startup_timeout) release notes |
| transformers | ≥ 4.48 (ModernBERT requirement) | answerdotai/ModernBERT-base model card; plan §2.1 |
| optimum | current (`optimum-cli export onnx`) | optimum quicktour (live, main branch). NOTE: the old spelling `optimum-cli export-onnx` is pre-1.6-era; current is `optimum-cli export onnx` |
| onnxruntime | ≥ 1.18 | pyproject `serve` extra; `quantize_dynamic`/`QuantType.QInt8` stable API |

## 2. Prerequisites

Only **HF_TOKEN** must be set for the happy path:

```bash
# repo tooling
uv sync --extra dev --extra deploy           # local venv (modal is deploy-time only)
modal setup                                  # one-time workspace link

# credentials: deploy-time env is read into the Modal Secret at deploy
export HF_TOKEN=hf_...                       # push-to-hub + dataset verification
```

Python 3.13 is used for local verification; the containers run **3.12** (`debian_slim(python_version="3.12")`, per the committed app).

## 3. Training app — deploy & run

```bash
# deploy (app definition + image build; no GPU cost while idle)
HF_TOKEN=... uv run --extra deploy modal deploy deploy/modal_app.py

# run with defaults: 5 epochs, batch 16, grad-accum 2, lr 2e-5, seed 42, eval-test
HF_TOKEN=... uv run --extra deploy modal run deploy/modal_app.py

# explicit run + Hub push
HF_TOKEN=... uv run --extra deploy modal run deploy/modal_app.py \
    --epochs 5 --batch-size 16 --grad-accum 2 --lr 2e-5 --seed 42 \
    --push-to-hub Lucius-Morningstar/mailroom-modernbert-classifier --eval-test
```

What happens inside the container:

1. `src/` + `training/` + `configs/` are bundled into the image at deploy time
   (`add_local_dir` → `/root/src`, `/root/training`, `/root/configs`);
   `PYTHONPATH=/root/src`, `MAILROOM_ML_ROOT=/root`.
2. The data pin from `src/mailroom_ml/config.py` is verified against the live
   Hub **before** the trainer starts (`HfApi().dataset_info(TRAINING_DATA_REPO,
   revision=TRAINING_DATA_REVISION)` — the pinned revision is the contract, a
   broken pin fails before any GPU minute) and exported to the container as
   `TRAINING_DATA_REVISION`.
3. `train_modernbert.py` runs via subprocess with the exact documented flags
   (`--data <repo> --output /checkpoints/latest --epochs … --batch-size …
   --grad-accum … --lr … --seed … [--push-to-hub <repo>] [--eval-test]`).
4. The trainer output lands at `/checkpoints/latest` on the
   **`modernbert-checkpoints`** Volume; on success the run is archived at
   `/checkpoints/runs/<run-id>/` and the Volume is committed once.

### Cost notes (L4)

- On-demand L4 is roughly **$0.5–0.6/h** (verify live: `modal billing rates`,
  1.5.4+ CLI) — the plan's budget is a one-time **~1–2 L4-hour run**. The job
  is scale-to-zero by default: no `keep_warm`, nothing billed between runs.
- The image build and the dataset/model download happen inside the 4 h function
  timeout (`startup_timeout` 10 min); the first boot pays ModernBERT weight +
  dataset download once (cached on the `mailroom-ml-hf-cache` Volume).
- A smoke run costs minutes: `modal run deploy/modal_app.py --epochs 1 --limit 64`
  (the `--limit` flag is a trainer flag; it still exercises the full path).

### Checkpoint persistence & rollback

| Path (inside the Volume) | Meaning |
| --- | --- |
| `/checkpoints/latest/` | stable pointer — what the export step consumes |
| `/checkpoints/runs/<run-id>/` | per-run archive, retained forever (rollback) |

- Rollback = point the downstream step at a previous `runs/<run-id>/`
  (`modal volume ls modernbert-checkpoints`, `modal volume get …`), or re-pull
  `CLASSIFIER_MODEL_REPO` from the Hub if the run was pushed.
- Failed runs never commit — `latest/` keeps the last good checkpoint, and the
  partial directory is discarded with the container.
- Copy a checkpoint locally:

```bash
mkdir -p artifacts/pytorch
modal volume get modernbert-checkpoints latest artifacts/pytorch/model
```

### `--push-to-hub` mechanics

`HF_TOKEN` (from the deploy-time env Secret) authorizes the trainer's
`--push-to-hub <repo>`. The trainer's own push uses the standard
`huggingface_hub`/transformers upload — `CLASSIFIER_MODEL_REPO` in
`src/mailroom_ml/config.py` is the operator-set publish target; the Modal app
never assumes it, the flag is explicit per run.

## 4. ONNX export + parity (the primary serving artifact)

Requires the train/serve extras (torch, transformers, onnxruntime). Exporting
with torch ≥ 2.14 additionally needs the ONNX toolchain wheels (`onnx`,
`onnxscript` — the 2.14 exporter imports them):

```bash
uv sync --extra train --extra serve
uv pip install onnx onnxscript        # torch >= 2.14 export/quantize toolchain

# 1) export: fp32 model.onnx + int8 model_quantized.onnx (dynamic batch+seq)
uv run python deploy/onnx_export.py \
    --pytorch-dir artifacts/pytorch/model --out-dir artifacts/onnx/model

# 2) parity gate
uv run python deploy/onnx_parity_check.py               # fp32 @ 1e-4 + int8 report
uv run python deploy/onnx_parity_check.py --require-int8  # also fail on int8 argmax flips
#    or as a pytest test (marker "serve", self-skips without artifacts)
uv run python -m pytest -m serve tests/test_deploy.py -v
```

Bundle layout produced:

```
artifacts/onnx/model/
    model.onnx             fp32  (graph inputs: input_ids, attention_mask int64,
                                  dynamic batch + sequence; one logits_<head>
                                  output per head)
    model_quantized.onnx   int8 dynamic-quantized (same contract)
    labels.json            label maps (self-contained serving + parity)
    tokenizer.json / tokenizer_config.json / config.json / export_meta.json
```

**Why not `optimum-cli export onnx`?** Both optimum paths
(`optimum-cli export onnx` and `ORTModelForSequenceClassification(export=True)`)
assume a standard `AutoModelForSequenceClassification` checkpoint. Our artifact
is a shared encoder + per-class conditional heads kept in `heads.pt` (outside
`config.json`) — exporting through either optimum path would initialize
**random head weights** and silently ship a broken graph. The robust route for
this architecture is `torch.onnx.export` with explicit dynamic axes, then
`onnxruntime.quantization.quantize_dynamic(…, weight_type=QuantType.QInt8)`
(the same quantizer optimum's `ORTQuantizer` wraps). If the model layer ever
publishes a standard loopback checkpoint, the documented optimum equivalent is:

```bash
optimum-cli export onnx --model <repo-or-dir> --task text-classification \
    --quantize int8 --dynamic-batchsize artifacts/onnx/model
```

**Exporter pin (`dynamo=False`)** — torch 2.14's `torch.onnx.export` defaults
to the Dynamo exporter, which currently emits an invalid graph for this
architecture (`Split` with a removed `num_outputs` attribute; `onnx.checker` and
onnxruntime both reject it). The export script pins the stable
TorchScript-tracer exporter and validates the graph (`onnx.checker` +
`onnxruntime.InferenceSession`) before quantizing. Revisit when Dynamo's Split
emission is fixed.

**Parity semantics (measured live on the real architecture, 2026-09-18):** the
fp32 `model.onnx` matched the PyTorch reference to ≤ 6e-6 on every head — the
**1e-4 gate is the export correctness contract**. Dynamic int8 weight
quantization perturbs logits (drift is weight-dependent; the measured fixture
drifted ~1.0 on logit scale with **random** heads); the int8 bundle is
therefore gated on **argmax agreement** via `--require-int8` (production gate)
with drift always reported — a trained checkpoint's confident margins survive
int8, and the calibration step downstream consumes probabilities, not raw
logits.

## 5. Fallback serving app (`mailroom-ml-serve`) — FALLBACK only

The PLAN's primary serving path is the **local ONNX CPU session**
(`artifacts/onnx/model/model_quantized.onnx`, ~147 MiB, ms loads, ≈0 $/doc).
The Modal app is an escape hatch (pipeline node that cannot host a local
session, demo endpoints). It runs the **same int8 artifact** on CPU — the image
deliberately excludes torch/transformers.

```bash
uv run --extra deploy --extra serve modal secret create mailroom-ml-serve-token \
    SERVE_API_TOKEN="$(openssl rand -hex 24)"
uv run --extra deploy --extra serve modal deploy deploy/serve_app.py
# artifacts/onnx/model is bundled at deploy when present; otherwise populate:
#   modal volume put mailroom-ml-onnx artifacts/onnx/model /model-vol

curl -s https://<workspace>--mailroom-ml-serve-health.modal.run          # metadata
curl -s -H "Authorization: Bearer $SERVE_API_TOKEN" -H "Content-Type: application/json" \
     -d '{"texts": ["Exhibit A: Software License Agreement …"]}' \
     https://<workspace>--mailroom-ml-serve-predict.modal.run
```

- `model_id` is passed at deploy via `SERVE_MODEL_ID` (default:
  `CLASSIFIER_MODEL_REPO`); the label maps ship inside the artifact bundle.
- Bearer token required on `/predict` (from the deploy-time env or the named
  `mailroom-ml-serve-token` secret) — never `unauthenticated=True` for this.
- Rollback: redeploy the previous app version (`modal app rollback
  mailroom-ml-serve`) or point `SERVE_MODEL_DIR` at an older Volume path.

## 6. Verification checklist (this repo)

```bash
uv run --python 3.13 python -c "import ast; ast.parse(open('deploy/modal_app.py').read())"
uv run --python 3.13 python -c "import ast; ast.parse(open('deploy/serve_app.py').read())"
uv run --extra dev --extra deploy python -m pytest tests/test_deploy.py -v
```

The core suite is green with **no** modal/torch/onnxruntime installed — every
deploy-layer test is `importorskip`/`skipif` guarded; network is never touched.
