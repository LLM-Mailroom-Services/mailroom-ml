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
| Modal SDK | **1.6.0** (2026-09-28) | <https://modal.com/docs/sdk/py/releases> (live, fetched 2026-10-02) + PyPI json (`pypi.org/pypi/modal/json` → 1.6.0, `requires_python <3.15,>=3.10`) |
| GPU config | string API, e.g. `gpu="L4"` | current docs examples (flux / gpu_fallbacks / llm_inference) — the committed app's "Modal ≥1.0 configures GPUs by string" comment is **confirmed correct**; `modal.gpu.L4()` objects are gone |
| Web endpoints | `@modal.fastapi_endpoint` | current Web Functions guide — `@modal.web_endpoint` was renamed to `fastapi_endpoint` prior to v0.73.82 |
| `App(name, tags=)`, `Image.debian_slim(python_version=)`, `uv_pip_install`, `add_local_dir`, `.env()`, `Secret.from_dict`, `Volume.from_name(create_if_missing=True)` + `commit()`, `startup_timeout=` | current | `modal.App`/`modal.Image`/`modal.Secret`/`modal.Volume` reference pages (live) + 1.2.0 (tags) and 1.1.4 (startup_timeout) release notes |
| transformers | ≥ 5.18 | pyproject `train` extra; ModernBERT model card requires ≥ 4.48 |
| optimum | ≥ 2.3 (`optimum-cli export onnx`) | optimum quicktour. NOTE: the old spelling `optimum-cli export-onnx` is pre-1.6-era; current is `optimum-cli export onnx` |
| onnxruntime | ≥ 1.30 | pyproject `serve` extra; `quantize_dynamic`/`QuantType.QInt8` stable API |

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
   (`--data <repo> --output /checkpoints/runs/<run-id> --epochs … --batch-size …
   --grad-accum … --lr … --seed … [--eval-test]`). The app does **not** pass
   `--push-to-hub`: a trainer-side push would upload before ONNX export/parity
   and even from a run where no epoch cleared the selection gate (see
   "`--push-to-hub` mechanics"). Smoke runs write to `/checkpoints/smoke-<ts>`
   and never touch `latest/` or the Hub.
4. On a successful **non-smoke** train (`#19`, default `--export-onnx`):
   1. `deploy/onnx_export.py` writes `onnx/` beside the run checkpoint
      (`torch.onnx.export`, never `optimum-cli`).
   2. `deploy/onnx_parity_check.py --require-int8` must pass.
   3. Only then is `/checkpoints/latest` promoted (copy + rename). Failed
      export/parity raises and **leaves `latest/` on the last good
      checkpoint**.
   4. Only then, and only if the trainer's selection gate was met
      (`summary.json` → `checkpoint_selection.gate_met`, i.e. `selected_epoch
      > 0`), the app uploads the run to the `--push-to-hub` repo.
   Skip the chain with `--no-export-onnx` on `spawn_train.py` / `modal run`
   (a run that skipped parity is **never** pushed to the Hub).
   A `--resume` that is already at/after `--epochs` trains nothing (the
   trainer's `summary.json` carries `nothing_trained: true`): the app skips
   export, promotion and push for it.
5. The run directory stays at `/checkpoints/runs/<run-id>/` on the
   **`modernbert-checkpoints`** Volume for rollback; the Volume is committed
   after a successful promote.

### Cost notes (L4)

- On-demand L4 is roughly **$0.5–0.6/h** (verify live: `modal billing rates`,
  1.5.4+ CLI) — the plan's budget is a one-time **~1–2 L4-hour run**. The job
  is scale-to-zero by default: no `keep_warm`, nothing billed between runs.
- The image build and the dataset/model download happen inside the 4 h function
  timeout (`startup_timeout` 10 min); the first boot pays ModernBERT weight +
  dataset download once (cached on the `mailroom-ml-hf-cache` Volume).
- A smoke run costs minutes: `uv run --extra deploy python deploy/spawn_train.py --smoke`
  (a 24-micro-batch run that exercises the full path; its timing feeds the
  budget guard, so it pins `--checkpoint-every` to the real run's cadence
  rather than inheriting its own `--log-every 4`, which would put a checkpoint
  save inside every 8 micro-batches of the measurement).

### Checkpoint persistence & rollback

| Path (inside the Volume) | Meaning |
| --- | --- |
| `/checkpoints/latest/` | stable pointer — what the export step consumes |
| `/checkpoints/runs/<run-id>/` | per-run archive, retained forever (rollback) |

- A cut run lives in `/checkpoints/runs/<run-id>/` — **not** `latest/`, which
  only ever holds the last *successful* run. Resume it with
  `spawn_train.py --resume /checkpoints/runs/<run-id> --epochs <original N>`;
  resuming `latest/` re-trains nothing.
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

`HF_TOKEN` (from the deploy-time env Secret) authorizes the upload. On Modal
the **app** pushes, not the trainer: `--push-to-hub <repo>` on `spawn_train.py`
/ `modal run` is held by `train()` and uploaded after ONNX export + parity
(and after `latest/` is promoted) — never from a smoke, never from a run whose
selection gate was not met (the uncalibrated final epoch), never from a resume
that trained nothing. The upload ignores `optimizer.pt`, `scheduler.pt`,
`resume.json` and the `onnx/` bundle (the model repo has only ever carried the
PyTorch checkpoint; the ONNX bundle stays on the Volume /
`artifacts/onnx/model`). `CLASSIFIER_MODEL_REPO` in `src/mailroom_ml/config.py`
is the operator-set publish target; the Modal app never assumes it, the flag is
explicit per run.

A **local** `train_modernbert.py --push-to-hub <repo>` still uploads from the
trainer, but refuses (exit 3) when `selected_epoch == 0`; pass `--push-ungated`
to publish such a checkpoint deliberately.

## 4. ONNX export + parity (the primary serving artifact)

Requires the train/serve extras (torch, transformers, onnxruntime, plus
`onnx` / `onnxscript` — the torch 2.14 exporter imports them):

```bash
uv sync --extra train --extra serve

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
    temperatures.json / summary.json / train_counts.json /
    ood_probe.json / routing_thresholds.json
                           serving sidecars, copied when the checkpoint has
                           them — `load_bundle` on this directory reads the
                           per-head temperatures, head-exclusion policy,
                           authentic-support counts, OOD probe and thresholds
                           from them (without them it loads at T=1 with no
                           exclusions / OOD / support)
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
logits. The int8 argmax/drift is measured against the **fp32 ONNX graph** (the
fp32 graph is already gated against PyTorch at 1e-4, so this isolates the
quantization error).

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
- No truncation (#85): a text over `SERVE_MAX_LENGTH` (8,192) tokens gets an
  HTTP 413 rather than being silently cut — send already-windowed text (the
  `mailroom_ml.windows` path). Requests are padded to their own longest text,
  not to 8,192. Head labels come from the bundle's `trainable_id2label`
  (routing-only labels such as doc_type `unknown` have no ONNX logit column).
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
