# Held-out-plus — full original plan, progress, and remaining work

**Tracker:** [mailroom-ml #40](https://github.com/LLM-Mailroom-Services/mailroom-ml/issues/40)  
**PR (tooling + handoff):** [#39](https://github.com/LLM-Mailroom-Services/mailroom-ml/pull/39)  
**Next-agent runbook:** [`HELDOUT-PLUS-EVAL-PICKUP.md`](./HELDOUT-PLUS-EVAL-PICKUP.md)  
**Pool report:** [`../reports/HOLDOUT-PLUS-V1-REPORT.md`](../reports/HOLDOUT-PLUS-V1-REPORT.md)  
**Last updated:** 2026-09-27 UTC · branch `cursor/heldout-plus-modal-hf-optional-dfc5` @ `36c2846+`

---

## 1. Original plan (do not shrink)

### 1.1 Objective (operator / thread goal)

Complete the **expanded ModernBERT held-out-plus test evaluation**:

1. Run **exactly one** **exios66 Modal L4** job over **1,323 documents**:
   - CLI: `training/eval_modernbert.py --subset heldout-plus --sample 0 --seed 42`
   - Checkpoint: **Arm B** on Modal volume `modernbert-checkpoints` → path **`latest`**
   - Run tag: **`m9a-local-20260927-014429`**
2. Produce **TEST-EVAL interpretation**, including explicit **323 vs 1,323** metric slices.
3. **No wasted GPU spend:** a single full pass only — no Modal smoke or partial `--sample N` runs on L4.

### 1.2 Budget / runtime (pre-registered)

- Operator approved ~**$0.16** / ~**12 min** wall from 323-doc Modal run scaling (~1,554 windows, L4 @ $0.80/hr).
- Pre-flight on CPU confirmed **1,323/1,323** docs window without error; inference **KeyError: '5'** (runner-up `unknown`) fixed before spending on full pool.

### 1.3 Relationship to M9a / #112

| Surface | n_docs | Role |
| --- | ---: | --- |
| Canonical held-out test | **323** | **#112 gates** (contract/correspondence macro-F1, doc_type acc, window ECE) — **already evaluated** for Arm B |
| Held-out-plus v1 | **1,323** | **Extended monitoring** (323 + 1,000 Enron correspondence); **does not replace** #112 gate surface |

Baseline for 323-vs-1,323 compare: `reports/eval_m9a-local-20260927-014429.json` (+ TEST-EVAL copies under `reports/TEST-EVAL/`).

### 1.4 Success criteria (completion audit)

| # | Deliverable | Proof |
| ---: | --- | --- |
| A | `reports/eval_m9a-local-20260927-014429-heldout-plus.json` on `main` | `"eval_subset": "heldout-plus"`, `"n_docs": 1323`, ~1554 windows |
| B | `reports/TEST-EVAL/TEST-EVAL-REPORT-heldout-plus-m9a-local-20260927-014429.md` | From `interpret_heldout_plus_eval.py` |
| C | `reports/TEST-EVAL/TEST-EVAL-COMPARE-323-vs-1323-m9a-local-20260927-014429.md` | Slice compare canonical vs full pool |
| D | `reports/TEST-EVAL/MANIFEST.json` | heldout-plus entry |
| E | Modal provenance | One L4 run; wall time / cost in JSON sidecar or commit evidence |

### 1.5 Explicitly out of scope (phase 2)

- Hub publish of `heldout-plus-v1` as versioned dataset (after clean GPU eval).
- Corporate-record top-up from new collection.
- CUAD-full contract mapping (41 → 25 subclasses) + SEC overlap audit.
- Re-running M9a **323-doc** gates (already done for Arm B).

---

## 2. Architecture (what the eval does)

- **Model:** ModernBERT intake classifier — shared encoder + conditional subclass heads (`mailroom_ml.inference.load_bundle`).
- **Windowing:** 8,192 tokens, 512 overlap, plurality merge (`mailroom_ml.windows`).
- **Held-out-plus load path:** `eval_modernbert._load_eval_docs` — canonical test from pinned corpus + `data/heldout_plus_v1/documents.parquet`; rejects filename overlap with canonical test.
- **Modal eval app:** `deploy/eval_app.py` — mounts checkpoint volume, optional bake of `data/heldout_plus_v1` into image at deploy; L4 GPU; invokes eval CLI.
- **One-shot runner:** `training/run_heldout_plus_modal_eval.sh` — pool audit → preflight → (optional) `modal deploy` → **one** `modal run` → post-check `n_docs==1323` → `interpret_heldout_plus_eval.py`.

---

## 3. Pool design (held-out-plus v1) — DONE (CPU)

**Total: 1,323 = 323 canonical test + 1,000 Enron correspondence (seed 42).**

| pool | n | source |
| --- | ---: | --- |
| canonical test | 323 | `mailroom-finetune` @ `19720ceb` (`split == test`) |
| plus v1 | 1,000 | `Lucius-Morningstar/enron-correspondence-dedup`, test split |
| **total** | **1,323** | `--subset heldout-plus` |

**Per doc_type (full pool):** contract 60 · corporate_record 47 · correspondence **1,085** · insurance_claim 114 · merger_agreement 17.

**Builder:** `training/build_heldout_plus.py` — join Enron blind/GT by filename, exclude train+val+test filenames (85 skipped), `deterministic_normalize`, leak audits → `gates_ok: true`.

**Preflight:** `training/preflight_heldout_plus.py` — 1,323/1,323 windowed, **1,554** windows, 0 errors.

**Local artifact:** `data/heldout_plus_v1/` (gitignored; rebuild commands in pickup doc).

Full lineage tables: `reports/HOLDOUT-PLUS-V1-REPORT.md`.

---

## 4. Progress chronology

### 4.1 M9a baseline (323-doc) — DONE before held-out-plus GPU goal

| Item | Status | Evidence |
| --- | --- | --- |
| Arm B train + 323-doc Modal/local eval | Done | `reports/eval_m9a-local-20260927-014429.json` |
| TEST-EVAL reports Arm A vs B | Done | `reports/TEST-EVAL/*` |
| Arm B doc_type acc (harness) | 0.9505 | eval JSON |
| #112 subclass gates on 323 | Fail (expected diagnosis) | `gates-check-armB.txt` |

### 4.2 Pool + inference fix — DONE (merged to `main`)

| When | What | Evidence |
| --- | --- | --- |
| 2026-09-27 | Held-out-plus v1 builder + 1,000 Enron extension | `training/build_heldout_plus.py`, commits `21c7d49`, `64b34ed`, `2ba0e15` |
| 2026-09-27 | Pre-flight 1323/1323, $0.16 projection | `training/preflight_heldout_plus.py`, report § |
| 2026-09-27 | **Inference fix:** runner-up `unknown` id 5 `KeyError` | `src/mailroom_ml/inference.py`, test `test_runner_up_unknown_with_trainable_map` |
| 2026-09-27 | `--subset heldout-plus` in eval CLI | `training/eval_modernbert.py` |
| 2026-09-27 | `eval_app` mounts plus dir when present | `deploy/eval_app.py` |

### 4.3 Tooling merged to `main` (PRs #35–#37)

| PR | Deliverable |
| --- | --- |
| **#35** | `training/run_heldout_plus_modal_eval.sh`, `interpret_heldout_plus_eval.py`, GHA workflow `heldout-plus-modal-eval`, fetch/build scripts |
| **#36** | Preflight fix, run bootstrap, heldout-plus wiring |
| **#37** | Public canonical corpus fallback in eval (HF optional for plus path), `secrets.env.example` |

`main` tip at handoff: **`838090b`**.

### 4.4 PR #39 (open) — runner ergonomics + results push + handoff

| Commit | Change |
| --- | --- |
| `afa5882` | Run script: **Modal tokens only**; HF optional |
| `d6cfc3d` | GHA: `push_results` → branch `heldout-plus-eval/<run_tag>` after `n_docs=1323` gate |
| `d62e946`–`36c2846` | `HELDOUT-PLUS-EVAL-PICKUP.md`, board card, **#40**, `AGENTS.md` links |

### 4.5 Cloud agent session `bc-01a0e2b1` — prep only, **no full L4**

| Done | Not done |
| --- | --- |
| Rebuilt/verified `data/heldout_plus_v1` on snapshot | `MODAL_TOKEN_*` never in pod |
| Preflight green | **0** `heldout-plus-modal-eval` workflow runs |
| HF download Arm B checkpoint; 1-doc CPU smoke (~3 min/doc) | No `eval_*-heldout-plus.json` anywhere on origin |
| Documented blockers; GHA dispatch **403**; browser not logged in | ~70 h CPU estimate for full 1,323 — **not** acceptable vs Modal L4 requirement |
| Issue **#40**, expanded handoff, timers to poll GHA | |

**Modal GPU spend for full 1,323-doc eval:** **$0** (intentional until credentials).

---

## 5. Remaining work (single critical path)

**One** of:

### Path A — New Cursor cloud agent + Modal secrets

Environment: https://cursor.com/dashboard/cloud-agents/environments/e/f88eb111-b84f-11f1-977f-f6b8f2fcf9b2  
Secrets: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` (exios66). **New agent boot required.**

```bash
./training/run_heldout_plus_modal_eval.sh
# RUN_TAG=m9a-local-20260927-014429 MODULE=latest (defaults)
```

### Path B — GitHub Actions (human, signed in)

Repo secrets: Modal required; HF optional.  
Actions → **heldout-plus-modal-eval** → branch **`cursor/heldout-plus-modal-hf-optional-dfc5`** → run defaults → `push_results: true`.

### Path C — Results already run elsewhere

Fetch `heldout-plus-eval/m9a-local-20260927-014429` or GHA artifact; merge to `main`.

**Then:** commit artifacts, close **#40** with SHAs + Modal wall/cost, merge **#39** if appropriate.

Detailed steps: [`HELDOUT-PLUS-EVAL-PICKUP.md`](./HELDOUT-PLUS-EVAL-PICKUP.md).

---

## 6. File map (quick reference)

| Path | Purpose |
| --- | --- |
| `training/build_heldout_plus.py` | Build 1,000-doc extension |
| `training/preflight_heldout_plus.py` | CPU window contract over 1,323 |
| `training/run_heldout_plus_modal_eval.sh` | One deploy + one Modal L4 run |
| `training/interpret_heldout_plus_eval.py` | TEST-EVAL + 323 vs 1,323 markdown |
| `deploy/eval_app.py` | Modal L4 eval app |
| `.github/workflows/heldout-plus-modal-eval.yml` | Manual GHA |
| `reports/HOLDOUT-PLUS-V1-REPORT.md` | Pool science + run table |
| `reports/eval_m9a-local-20260927-014429.json` | **323-doc baseline** |
| `governance/HELDOUT-PLUS-EVAL-PICKUP.md` | Next-agent checklist |

---

## 7. PR merge list (for next agent)

| PR | State | Action |
| --- | --- | --- |
| [#39](https://github.com/LLM-Mailroom-Services/mailroom-ml/pull/39) | Open | Merge after review (or dispatch GHA from branch before merge) |
| #35–#37 | Merged | On `main` |

---

## 8. Issue closure checklist (#40)

- [ ] Full plan §1.4 criteria A–E satisfied on `main`
- [ ] Comment on #40: commit SHAs, Modal run id/wall/cost, link to TEST-EVAL compare
- [ ] Optional: comment on mailroom-issues #112 that extended 1,323 monitoring eval landed (not a gate rerun)
