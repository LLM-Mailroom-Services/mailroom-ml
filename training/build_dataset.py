#!/usr/bin/env python3
"""CLI: stage + publish the ModernBERT training set to HuggingFace.

Port of ``modernbert/prep.py`` + ``modernbert/publish.py`` (Mailroom-Corpus-
EDA commit cf096fa), re-sourced to the working corpus copy
``Lucius-Morningstar/mailroom-finetune`` @ ``FINETUNE_REVISION``.

Usage:
    uv run python training/build_dataset.py --stage-only
    uv run python training/build_dataset.py --publish          # create + upload + verify
    uv run python training/build_dataset.py --publish --repo-id Lucius-Morningstar/mailroom-modernbert-training
    uv run python training/build_dataset.py --publish --no-stage   # upload an
        # already-assembled tree (adopted enrichment): skips stage() so the
        # enrichment-inclusive labels.json / manifest.txt / dataset_info.json
        # are published as-is instead of being overwritten canonical-only.

Stage-only by default; ``--publish`` is OPERATOR-ONLY (never automatic):
it creates/updates the public dataset repo, uploads the verified stage tree
via ``huggingface_hub.HfApi`` directly (no ad-hoc upload code, clear commit
messages), then byte-verifies the sidecars' sha256s against the Hub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# training/ scripts sit outside the package — anchor on the repo root (.git)
# and put src/ on sys.path so the CLI runs from any checkout (mirrors the
# committed publish.py bootstrap).
_b = Path(__file__).resolve()
while not (_b / ".git").is_dir():
    _parent = _b.parent
    if _parent == _b:
        raise SystemExit(
            f"repo root not found: no .git directory above {Path(__file__).resolve()}")
    _b = _parent
ROOT = _b
sys.path.insert(0, str(ROOT / "src"))

from mailroom_ml import config as cfg  # noqa: E402
from mailroom_ml.dataset import (  # noqa: E402
    has_adopted_enrichment,
    refresh_dataset_info,
    stage,
    verify_stage,
)
from mailroom_ml.labels import DOC_TYPES, SUBCLASS_BY_CLASS  # noqa: E402


def render_card(stats: dict, repo_id: str, enriched: bool = False) -> str:
    """Dataset-card markdown — documents the FULL synthetic-data fine-tuning
    layer, not just this repo's rows: corpus → finetune working copy →
    (gated) enrichment tiers → build/publish → trainer consumption.

    ``enriched`` is set by the publish CLI when the stage carries adopted
    enrichment parquet — the card must not claim "pure-canonical baseline"
    over an enrichment-inclusive tree (doc-currency law).
    """
    counts = stats["counts"]
    docs = counts["documents"]
    wins = counts.get("windows", {})
    n_train = docs.get("train", 0)
    n_val = docs.get("validation", 0)
    n_test = docs.get("test", 0)
    if enriched:
        enrichment_note = (
            "Synthetic enrichment (tier 1/2/3) is assembled by\n"
            "`training/assemble_enrichment.py` from external pools (enron, cuad,\n"
            "cms, gnotheia, bdr, insurbias, blind) with per-tier caps, a 7-gate\n"
            "audit, and `example_weight`/`lineage`/`tier` columns. **This\n"
            "revision ADOPTS tier-1 source-matched enrichment** (CUAD contract\n"
            "expansion + subclass-balanced Enron correspondence). Adopted rows\n"
            "land in `documents/train` + `windows/train` ONLY — validation and\n"
            "the held-out test split are never touched (ladder rule). Per-row\n"
            "provenance is in `enrichment_provenance.parquet`; every rejected or\n"
            "cut row is in `enrichment_audit.jsonl`; the manifest's\n"
            "`# ---- enrichment ----` block records the pools, caps and counts."
        )
    else:
        enrichment_note = (
            "Synthetic enrichment (tier 1/2/3) is assembled by\n"
            "`training/assemble_enrichment.py` from external pools (enron, cuad,\n"
            "cms, gnotheia, bdr, insurbias, blind) with per-tier caps, a 7-gate\n"
            "audit, and `example_weight`/`lineage`/`tier` columns. **No\n"
            "enrichment tier is adopted into this dataset yet** — adoption is\n"
            "gated by the plan's §6.1 A/B ladder (an adopted tier must beat the\n"
            "no-enrichment baseline on the held-out test split before it ships\n"
            "here). This revision is the pure-canonical baseline."
        )
    return f"""---
license: cc-by-4.0
language:
- en
task_categories:
- text-classification
tags:
- legal
- mailroom
- modernbert
- hierarchical-classification
- synthetic-data
pretty_name: mailroom-modernbert-training
configs:
- config_name: documents
  default: true
  data_files:
  - split: train
    path: data/documents/train/*
  - split: validation
    path: data/documents/validation/*
  - split: test
    path: data/documents/test/*
- config_name: windows
  data_files:
  - split: train
    path: data/windows/train/*
  - split: validation
    path: data/windows/validation/*
---

# mailroom-modernbert-training

Cleaned + prepared hierarchical-classification training set for the
**ModernBERT-base** ingest fast-path — the fine-tuning surface of the
mailroom-ml synthetic-data layer.

## The layer (end to end)

```
Lucius-Morningstar/mailroom-dataset      corpus (GT labels, canonical v9)
        │  (working copy, pinned)
        ▼
Lucius-Morningstar/mailroom-finetune     corpus snapshot @ {cfg.FINETUNE_REVISION[:8]}
        │  training/build_dataset.py (stage + verify)
        ▼
THIS REPO (mailroom-modernbert-training) @ pinned revision
        │  training/train_modernbert.py (--data <repo>)
        ▼
Lucius-Morningstar/mailroom-modernbert-classifier   (trained checkpoint)
```

{enrichment_note}

## Rows

| | |
|---|---|
| Documents | {n_train + n_val + n_test} (train {n_train} / validation {n_val} / test {n_test}) |
| Windows (8,192-token) | {sum(wins.values())} (train {wins.get('train', 0)} / validation {wins.get('validation', 0)}) |
| doc_type classes | {len(DOC_TYPES)} (+ `unknown`, inference-only) |
| Subclass heads | {len(SUBCLASS_BY_CLASS)} per-class heads |
| Model target | {cfg.MODEL_ID} (max_tokens={cfg.MAX_TOKENS}, overlap={cfg.WINDOW_OVERLAP_TOKENS}) |
| Source revision | `{cfg.FINETUNE_REVISION}` (GT-closure) |

## Configs

- **`documents`** — one row per document: `filename`, `document_id`,
  `content_sha256`, `source_revision`, `title`, `doc_text`, `doc_type`,
  `subclass`, `corpus_split`, `split` (train/validation/test),
  `token_estimate`. The held-out **test split (323 rows) never touches
  training** — it is the eval surface.
- **`windows`** — one row per pre-computed 8,192-token ModernBERT window
  (`title + "\\n\\n" + window`, 512-token overlap) for **train + validation
  only** — the fine-tune surface. Test stays document-level.

## Labels

- `doc_type` — 5 classes: {", ".join(DOC_TYPES)}.
- `subclass` — canonical key per class, normalized through the same mapping
  the sandbox corpus uses (`llm_dojo_scoring.corpus.normalize_corpus_subclass`,
  DMR-066; vendored in `mailroom_ml/labels.py`): `Service`/`service` unify,
  `Co_Branding` → `co_branding`, `Joint Venture _ Filing` → `joint_venture`.
  Per-class vocabularies in `vocabularies.json`; per-head id2label + class
  weights (inverse-frequency over the train split) in `labels.json`.

## Splits

- Corpus `train` (2,979) → **90/10 stratified** train/validation (stratified
  by `doc_type`, seed 42, deterministic `RandomState` shuffle — no sklearn
  dependency, byte-identical rebuilds).
- Corpus `test` (323) → held out entirely.
- Adopted tier-1 enrichment rows (when this revision is enriched) are
  appended to **train only** — validation and test are never touched.

## Input construction

`title` follows the corpus's title-wins convention: **subject →
exhibit_description → empty** (semantic-only). The filename fallback was
removed on 2026-09-20: it leaked the label (42.8% of filenames carried the
subclass token; 65.9% of rows had `title == filename`) and a real pipeline
filename does not encode the class. Each window input is
`title + "\\n\\n" + window_text` (body-only when there is no semantic
title), truncated to 8,192 tokens.

Text is run through the pipeline's **deterministic intake clerk**
(`deterministic_normalize`, vendored in `mailroom_ml/normalize.py`) at build
time — the same clerk `llm-mailroom apply_intake` runs before the classifier —
so training input is byte-representative of inference input (NFC, newline
unify, NBSP, zero-width strip, C0 controls, hyphen unwrap, blank-run collapse,
horizontal collapse, edge trim). A `filename_leak_audit` gate in
`verify_stage` fails the build if a filename-derived title ever returns.

## Provenance

Built by `mailroom_ml` (src/) + `training/build_dataset.py` in the
mailroom-ml repo. `manifest.txt` carries the build facts + sha256s; rebuilds
are byte-identical (sorted rows, seeded split, pinned tokenizer, no
timestamps in artifacts). Publishing is operator-only (`--publish`) and
byte-verifies every sidecar against the Hub after upload.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--publish", action="store_true",
                    help="create the dataset repo + upload + verify (operator-only)")
    ap.add_argument("--repo-id", default=cfg.TRAINING_DATA_REPO)
    ap.add_argument("--stage-only", action="store_true",
                    help="build the staged tree (default; --publish not given)")
    ap.add_argument("--no-windows", action="store_true",
                    help="skip the windows config (documents only)")
    ap.add_argument("--no-stage", action="store_true",
                    help="publish an ALREADY-assembled stage tree as-is (skip "
                         "stage(); required when the tree carries adopted "
                         "enrichment — stage() would rewrite the sidecars "
                         "canonical-only over it). Requires --publish.")
    args = ap.parse_args(argv)

    if args.no_stage:
        if not args.publish:
            print("ERROR: --no-stage only affects the publish path — add "
                  "--publish (there is nothing to do otherwise)", file=sys.stderr)
            return 2
        if not (cfg.STAGE_DIR / "manifest.txt").exists():
            print(f"ERROR: --no-stage: no staged tree at {cfg.STAGE_DIR} — run "
                  "training/build_dataset.py --stage-only first", file=sys.stderr)
            return 2
        # refresh dataset_info.json from the FULL tree (canonical stage() +
        # adopted enrichment parquet) before upload — assemble_enrichment.py
        # adds rows without touching it.
        enriched = has_adopted_enrichment(cfg.STAGE_DIR)
        stats = refresh_dataset_info(cfg.STAGE_DIR)
        print(f"using existing stage (no re-stage): "
              f"{json.dumps(stats['counts'], sort_keys=True)}")
    else:
        enriched = has_adopted_enrichment(cfg.STAGE_DIR)
        if enriched and args.publish:
            print(
                "ERROR: the stage carries adopted enrichment but --publish "
                "would re-run stage(), rewriting labels.json / dataset_info.json "
                "/ manifest.txt canonical-only while the enrichment parquet "
                "stays — an inconsistent publish. Re-run with: "
                "--publish --no-stage", file=sys.stderr)
            return 2
        try:
            stats = stage(cfg.STAGE_DIR, with_windows=not args.no_windows)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"staged: {json.dumps(stats['counts'], sort_keys=True)}")
        print(f"manifest sha256: {stats['manifest_sha256']}")
        if enriched:
            print("WARNING: stage carried enrichment parquet; re-staged the "
                  "canonical sidecars over it. Run "
                  "training/assemble_enrichment.py to regenerate, or publish "
                  "an enriched tree with --publish --no-stage.", file=sys.stderr)

    check = verify_stage(cfg.STAGE_DIR)
    if not check["ok"]:
        print("VERIFY FAILED:")
        for p in check["problems"]:
            print(f"  - {p}")
        return 1
    print(f"verify ok: {check['rows']} rows, splits {check['splits']}")

    if not args.publish:
        print("stage-only (pass --publish to upload to HF)")
        return 0

    # ---- operator-only publish path -------------------------------------
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                    private=False, exist_ok=True)
    print(f"repo ready: https://huggingface.co/datasets/{args.repo_id}")

    (cfg.STAGE_DIR / "README.md").write_text(
        render_card(stats, args.repo_id, enriched=enriched), encoding="utf-8")

    commit = api.upload_folder(
        folder_path=str(cfg.STAGE_DIR),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=(
            f"ModernBERT training set from {cfg.FINETUNE_REPO} @ "
            f"{cfg.FINETUNE_REVISION[:8]} "
            f"(documents {stats['counts']['documents']}, "
            f"windows {stats['counts'].get('windows', {})})"
        ),
    )
    print(f"uploaded: {commit.commit_url}")
    commit_sha = commit.commit_url.rstrip("/").rsplit("/", 1)[-1]

    print("uploaded; verifying sha256s...")
    results = []
    for rel in ("manifest.txt", "labels.json", "vocabularies.json",
                "README.md", "dataset_info.json"):
        local = cfg.STAGE_DIR / rel
        if not local.exists():
            continue  # e.g. --no-windows stage has no windows config
        hub_path = hf_hub_download(
            repo_id=args.repo_id, filename=rel, repo_type="dataset",
            revision=commit_sha)
        hub = Path(hub_path).read_bytes()
        verified = hub == local.read_bytes()
        results.append({
            "file": rel, "verified": verified,
            "local_sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
            "hub_sha256": hashlib.sha256(hub).hexdigest(),
        })
        print(f"  {rel}: {'OK' if verified else 'MISMATCH'}")
    if not all(r["verified"] for r in results):
        print("WARNING: some files could not be byte-verified against the Hub")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
