# Held-out-plus v1 — expanded ModernBERT test set (1,323 docs)

Built **2026-09-27**, CPU-only, no GPU spend. Canonical 323-doc held-out test
plus **1,000 fresh correspondence documents** from the Enron deduplicated
feeder corpus, cleaned through the exact training path and gated by the same
leak audits.

## Composition

| pool | n | source |
| --- | ---: | --- |
| canonical test | 323 | `mailroom-finetune` @ `19720ceb` (`split == test`) |
| plus v1 | 1,000 | `Lucius-Morningstar/enron-correspondence-dedup`, test split |
| **total** | **1,323** | `training/eval_modernbert.py --subset heldout-plus` |

Per-doc_type: contract 60 · corporate_record 47 · correspondence **1,085** ·
insurance_claim 114 · merger_agreement 17. Plus-subclass mix: email 981 ·
notice 7 · memo 6 · letter 5 · press_release 1 (the Enron test draw as drawn,
seed 42 — not rebalanced, so the extension measures the natural feeder mix).

## Feeder survey (why v1 is correspondence-only)

| candidate pool | rows | fresh vs train+val+test | verdict |
| --- | ---: | ---: | --- |
| Enron dedup test (correspondence) | 24,951 | **24,866** filename-fresh, 0 content-sha collisions in the 1,000-draw | **used** |
| CMS DE-SynPUF train (insurance) | 375 | 332 filenames already in train+val | absorbed — nothing fresh |
| CMS DE-SynPUF test (insurance) | 25 | 22 in train, 3 in test | absorbed — nothing fresh |
| mailroom-corpus v8 (all types) | 2,000 | **0** filename-fresh (fully absorbed into v9) | absorbed — nothing fresh |
| CUAD-full (contract, 510) | 510 | unchecked | **phase 2** — clause-level labels need a 41-category → 25-subclass mapping + SEC-exhibit content-overlap check |

Insurance stays at the canonical 114 rows and corporate_record at 47 until
new documents are sourced: no fresh pool for either exists on the Hub today.
Corporate-record top-up is new-collection work (acceptance: filename + content
exclusion vs train+val+test, `filename_leak_audit` clean, subclass in the
10-key observed set).

## Cleaning (training path, verbatim)

Builder: `training/build_heldout_plus.py` (deterministic, seed 42).

1. Join Enron `blind/test.jsonl` (subject + text) to `ground_truth/test.jsonl`
   (expected + expected_subclass) **by filename, never positionally**.
2. Exclude filenames present in train+val or canonical test (85 skipped).
3. Exclude content-sha matches under a different filename (`dedup_by_sha`
   semantics; 0 skipped in the draw).
4. `deterministic_normalize` on body; title = subject → exhibit_description →
   `""` (never filename — the 2026-09-20 leak fix). 964/1,000 rows carry a
   real subject; 36 are body-only.
5. `normalize_subclass` validation; empty bodies dropped (0).

## Leak gates (all green, `gates_ok: true`)

| gate | result |
| --- | --- |
| `title_eq_filename` / `title_looks_like_filename` | **0 / 0** (`clean: true`) |
| cross-split filename duplicates in selection | **none** |
| content-sha collisions vs train+val | **0** |
| title contains doc_type | 0 |
| title contains subclass | 9 (real subjects naming their topic — signal, not defect) |
| body contains subclass token | 300 (ordinary email diction — signal, not defect) |

Full audit: `data/heldout_plus_v1/audit.json` (local build artifact, gitignored;
`data/` never commits — re-run the builder to reproduce byte-identically).

## Plus-set quality mix (2026-09-27, local CPU)

| property | value |
| --- | --- |
| body chars min / median / max | 4 / 699 / 211,263 |
| token estimate min / median / max | 1 / 174 / 52,816 |
| nonempty subjects | 964 / 1,000 (932 unique) |
| folded-duplicate subject groups | 11 groups / 46 rows (top: bare `RE:`/`Re:` subjects as published — source trait, not a cleaning defect) |
| subclass mix | email 981 · notice 7 · memo 6 · letter 5 · press_release 1 |

Median 174 tokens keeps the extension firmly in the single-window regime
(pre-flight: 1,269/1,323 single-window overall); the long tail exercises the
plurality-merge path. The builder now also emits `token_estimate`, so plus
rows match the `build_documents` column contract exactly.

## Eval wiring

- `training/eval_modernbert.py --subset heldout-plus` loads canonical test +
  `data/heldout_plus_v1/documents.parquet`, rejects on any canonical overlap,
  reports `eval_subset: heldout-plus` (1,323 docs, all `split == test`).
- `deploy/eval_app.py` mounts the plus dir into the Modal image when present
  (fresh checkouts without the local build keep working for other subsets).
- CPU smoke (this build, local checkpoint, `--sample 1`, 5 docs, rc=0):
  doc_type acc **0.80**, conditional subclass **0.75** — the single miss is the
  known contract→corporate_record confusion also present in the 323 harness;
  the drawn Enron doc (`nemec-g/…`) classified correspondence→correspondence.

## Blocker fixed on the way here

The 1,000-doc `subset=all` Modal run (exios66, 2026-09-27) crashed in
`classify_windows` with `KeyError: '5'`: the runner-up argmax runs over the
**full** calibrated width, so inference-only `unknown` (id 5) is a legitimate
runner-up, but it was looked up in the **trainable-only** map (ids 0–4).
`src/mailroom_ml/inference.py` now resolves the abstain path and runner-up
label through the full `id2label`; regression test
`test_runner_up_unknown_with_trainable_map` fails on the old code with the
exact production traceback line and passes on the new (36/36 inference tests,
51/51 with the eval-CLI suite).

## Pre-flight validation (2026-09-27, local CPU)

Full-pool contract check over all **1,323** docs (labels, emptiness, title
leak, `window_document` at 8,192/512):

| check | result |
| --- | --- |
| windowed without error | **1,323 / 1,323** |
| label mismatches (`normalize_subclass`) | 0 |
| empty bodies | 0 |
| `title == filename` | 0 |
| single-window / multi-window docs | 1,269 / 54 (max 13 windows) |
| total windows | **1,554** |

Runtime projection from the 323-doc Modal run (541 windows in ~200 s remote
wall incl. stage pull + weight load): eval time scales ~linearly in windows,
so the full 1,323-doc run should land **under ~12 min wall ≈ $0.16** at
$0.80/hr L4. The pool will not crash or fail-open on windowing.

## Cost / runtime note (no new GPU billing this turn)

- This turn: **$0 Modal** (one failed L4 run from the prior session surfaced
  the bug above; all diagnosis, builds, and the 5-doc smoke ran on local CPU —
  the smoke took ~10.7 min wall for 5 docs).
- Projection for the full 1,323-doc L4 eval (from the 323-doc Modal run:
  ~0.6 s/doc remote): **~13–15 min ≈ $0.17–0.20** at $0.80/hr. Awaiting
  approval before spending it; the set is built and the crash is fixed, so the
  next run should complete in one attempt.

## Run the one-shot GPU eval (~$0.16)

**Do not** run smoke or partial samples — one full pass only.

| Path | Command |
| --- | --- |
| **exios66 shell** | `export HF_TOKEN=… MODAL_TOKEN_ID=… MODAL_TOKEN_SECRET=…` then `./training/run_heldout_plus_modal_eval.sh` (optional `SKIP_DEPLOY=1` if `eval_app` already baked the current `data/heldout_plus_v1`) |
| **GitHub Actions** | Repo → Actions → **heldout-plus-modal-eval** → Run workflow (requires the same three secrets in repo settings) |

Outputs: `reports/eval_<run_tag>-heldout-plus.json` plus `reports/TEST-EVAL/TEST-EVAL-REPORT-heldout-plus-*.md`.

## Phase 2 (tracked, not started)

1. Full 1,323-doc Modal eval on exios66 + interpretive report — **script/workflow above**; pending one authorized run.
2. Corporate-record top-up from newly sourced documents (acceptance criteria above).
3. CUAD-full contract mapping (41 clause categories → 25 subclasses) + SEC-exhibit
   content-overlap check vs training contracts.
4. Publish `heldout-plus-v1` as a versioned Hub dataset once the set is validated
   by a clean GPU eval.
