# Enrichment → publish → train runbook

Operator sequence for changing the **staged training set** on the Hub. Cross-links: mailroom-issues **#85**, mailroom-ml **#24**.

## Artifacts

| Artifact | Pin location | Hub repo |
| --- | --- | --- |
| Working corpus | `FINETUNE_REVISION` | `Lucius-Morningstar/mailroom-finetune` |
| Training set | `TRAINING_DATA_REVISION` | `Lucius-Morningstar/mailroom-modernbert-training` |
| Classifier | operator per run | `CLASSIFIER_MODEL_REPO` (optional `--push-to-hub`) |

## 1. Preflight

```bash
uv run python training/preflight.py
uv run python training/preflight.py --online   # HF_TOKEN set
```

## 2. Stage canonical tree (if not already present)

```bash
uv run python training/build_dataset.py --stage-only
```

## 3. Assemble enrichment (train split only)

```bash
uv run python training/assemble_enrichment.py \
  --stage data/modernbert_training/stage --tiers 1 --dry-run
# remove --dry-run when counts/audit look correct
```

Inspect `enrichment_audit.jsonl` and `manifest.txt` (`# ---- enrichment ----` block).

## 4. Publish

**Enrichment-inclusive tree** (do not re-run `stage()` over adopted rows):

```bash
uv run python training/build_dataset.py --publish --no-stage \
  --repo-id Lucius-Morningstar/mailroom-modernbert-training
```

Byte-verify sidecars must pass (#120).

## 5. Bump pin

Merge a commit updating `TRAINING_DATA_REVISION` in `src/mailroom_ml/config.py` to the new dataset revision **before** any Modal train.

## 6. Train

```bash
HF_TOKEN=... uv run --extra deploy python deploy/spawn_train.py --smoke
# full run only after operator budget sign-off (M9a: mailroom-issues #112, mailroom-ml #13)
```

Modal verifies the pin against the live Hub before GPU minutes.

## Rollback

- Hub: use a prior dataset revision; revert `TRAINING_DATA_REVISION`.
- Checkpoints: `modernbert-checkpoints` Volume `runs/<run-id>/` (see `deploy/README.md`).
