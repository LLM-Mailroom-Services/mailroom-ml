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
uv run python training/train/preflight.py
uv run python training/train/preflight.py --online   # HF_TOKEN set
```

## 2. Stage canonical tree (if not already present)

```bash
uv run python training/dataset/mailroom-dataset/build_dataset.py --stage-only
```

## 3. Assemble enrichment (train split only)

```bash
uv run python training/dataset/mailroom-dataset/assemble_enrichment.py \
  --stage data/modernbert_training/stage --tiers 1 --dry-run
# remove --dry-run when counts/audit look correct

# Tier-1 MAUD / S1 (#16): no Hub pin yet — pass a local JSONL/parquet
uv run python training/dataset/mailroom-dataset/assemble_enrichment.py --tiers 1 --dry-run \
  --maud-pool path/to/maud.jsonl --s1-pool path/to/s1.jsonl
```

Inspect `enrichment_audit.jsonl` and `manifest.txt` (`# ---- enrichment ----` block).

### Tier-2 blind pool (#26)

`--blind-pool` defaults to the pinned Enron blind **text** source
(`enron-correspondence-dedup` @ `BLIND_POOL_REVISION`, config `blind`). That
pin does **not** carry `doc_type_conf` / `subclass_conf` / `agreement`. Score
it first (does not invent confidences):

```bash
uv run python training/eval/score_blind_pool.py \
  --pool data/enrichment/enron_blind.parquet \
  --checkpoint artifacts/pytorch/model \
  --out data/enrichment/blind_scored.parquet
uv run python training/dataset/mailroom-dataset/assemble_enrichment.py --tiers 2 \
  --blind-pool data/enrichment/blind_scored.parquet --dry-run
```

Missing score columns abort with exit 2 (no silent adopt).

### Tier-3 label-card synthesis (#28)

All seven gates are implemented (rule cues, independent adjudication,
n-gram diversity, model disagreement). In-repo assembly never calls a live
LLM — operators supply `--tier3-cards` + `--tier3-candidates`. Failures land
in `enrichment_audit.jsonl` with stable reason codes. Prompt template id is
`TIER3_PROMPT_TEMPLATE_ID` (`label_card_v1`); `TIER3_GENERATOR_MODEL` is
empty until an operator pins a generator.

## 4. Publish

**Enrichment-inclusive tree** (do not re-run `stage()` over adopted rows):

```bash
uv run python training/dataset/mailroom-dataset/build_dataset.py --publish --no-stage \
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
