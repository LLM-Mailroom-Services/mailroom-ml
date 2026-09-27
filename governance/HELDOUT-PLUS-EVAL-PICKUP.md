# Held-out-plus eval — next-agent pickup (1,323-doc Modal L4)

> **Full original plan + progress log:** [`HELDOUT-PLUS-PLAN-AND-PROGRESS.md`](./HELDOUT-PLUS-PLAN-AND-PROGRESS.md) (canonical; mirrored in [#40](https://github.com/LLM-Mailroom-Services/mailroom-ml/issues/40) and [#39](https://github.com/LLM-Mailroom-Services/mailroom-ml/pull/39)).

**Status:** BLOCKED on credentials / manual GHA dispatch (2026-09-27 UTC; re-verified same pod `bc-01a0e2b1`).  
**Parent context:** mailroom-issues **#112** (M9a); extended monitoring pool, **not** the canonical 323-doc #112 gates.  
**Cloud agent run that prepared this handoff:** `bc-01a0e2b1-55b8-787b-a518-e08c7ce8dfc5`  
**PR (merge before or dispatch from):** https://github.com/LLM-Mailroom-Services/mailroom-ml/pull/39  
**Dedicated tracker:** https://github.com/LLM-Mailroom-Services/mailroom-ml/issues/40

---

## Mission (do not shrink)

Run **exactly one** exios66 **Modal L4** eval:

```text
training/eval_modernbert.py --subset heldout-plus --sample 0 --seed 42
```

- **Checkpoint:** Arm B on volume `modernbert-checkpoints` → **`module=latest`** (run tag `m9a-local-20260927-014429`).
- **Documents:** **1,323** (323 canonical test + 1,000 Enron correspondence extension).
- **No** smoke / partial sample Modal runs (operator approved ~**$0.16** / ~12 min wall).

Then produce **TEST-EVAL interpretation** including **323 vs 1,323** slices and update `reports/TEST-EVAL/MANIFEST.json`.

---

## Done when (verify every item)

| # | Evidence |
|---|----------|
| 1 | `reports/eval_m9a-local-20260927-014429-heldout-plus.json` exists in git on `main` (or merged from results branch below) |
| 2 | JSON: `"eval_subset": "heldout-plus"`, `"n_docs": 1323`, ~**1554** windows in `window_calibration.n_windows` |
| 3 | `reports/TEST-EVAL/TEST-EVAL-REPORT-heldout-plus-m9a-local-20260927-014429.md` |
| 4 | `reports/TEST-EVAL/TEST-EVAL-COMPARE-323-vs-1323-m9a-local-20260927-014429.md` |
| 5 | `reports/TEST-EVAL/MANIFEST.json` includes heldout-plus entry |
| 6 | Baseline for compare: `reports/eval_m9a-local-20260927-014429.json` (323-doc, already in repo) |

---

## What is already shipped (no redo unless broken)

### CPU / local (this VM snapshot)

- Pool: `data/heldout_plus_v1/` — `audit.json` → `gates_ok: true`, `selected: 1000`, seed 42 (**gitignored**; rebuild below if missing).
- Preflight (2026-09-27): `uv run python training/preflight_heldout_plus.py` → **1323 docs, 1554 windows**, `preflight heldout-plus OK`.
- Environment setup actions requested for `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` (+ optional `HF_TOKEN`) on environment `f88eb111-b84f-11f1-977f-f6b8f2fcf9b2` — **start a new agent after saving secrets**.
- Baseline 323 eval: `reports/eval_m9a-local-20260927-014429.json`.

### Code on branch `cursor/heldout-plus-modal-hf-optional-dfc5` (PR #39)

- `training/run_heldout_plus_modal_eval.sh` — one deploy + one `modal run`; **Modal tokens required**, **HF_TOKEN optional**.
- `.github/workflows/heldout-plus-modal-eval.yml` — manual dispatch; on success can **push** to `heldout-plus-eval/<run_tag>`.
- `training/interpret_heldout_plus_eval.py` — called by run script post-eval.
- `reports/HOLDOUT-PLUS-V1-REPORT.md` — pool lineage + run paths.
- `AGENTS.md` — Cursor Cloud secrets + “new agent required” note.

### On `main` (already merged #35–#37)

- Eval CLI `--subset heldout-plus`, public corpus fallback in `eval_modernbert.py`.
- `deploy/eval_app.py` bakes `data/heldout_plus_v1` when present at deploy time.
- Inference fix: runner-up `unknown` (`KeyError: '5'`) — required for long-tail docs.

### Not done

- **Zero** Modal GPU spend for full 1,323 run in agent sessions.
- **Zero** `heldout-plus-modal-eval` workflow runs (as of handoff).
- **No** `heldout-plus-eval/*` results branch on origin.

---

## Why the prior cloud agent could not finish

1. **`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`** never injected into pod `bc-01a0e2b1` (secrets apply at **new agent boot** only).
2. **GHA `workflow_dispatch`** returns **403** for the Cursor integration token — agent cannot click “Run workflow”.
3. **Browser** on VM: GitHub **not signed in** — no “Run workflow” button.
4. **CPU fallback** for 1,323 docs ≈ **70 h** on this VM (no GPU); does **not** satisfy “exios66 Modal L4” requirement.

---

## Path A — New Cursor cloud agent (preferred if exios66 Modal tokens)

1. **Environment:** https://cursor.com/dashboard/cloud-agents/environments/e/f88eb111-b84f-11f1-977f-f6b8f2fcf9b2  
   Add secrets: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` (exios66). Optional: `HF_TOKEN`.
2. **Start a new agent** on `mailroom-ml` (do **not** resume `bc-01a0e2b1` expecting secrets).
3. Checkout **`cursor/heldout-plus-modal-hf-optional-dfc5`** or merge **PR #39** first.
4. Rebuild pool if needed:

   ```bash
   uv sync --extra dev --extra deploy
   uv run python training/fetch_corpus.py
   uv run python training/fetch_enron_heldout_inputs.py
   uv run python training/build_heldout_plus.py \
     --enron-gt data/enrichment/enron_heldout/ground_truth/test.jsonl \
     --enron-blind data/enrichment/enron_heldout/blind/test.jsonl \
     --n 1000 --seed 42 --out data/heldout_plus_v1
   uv run python training/preflight_heldout_plus.py
   ```

5. **One-shot GPU:**

   ```bash
   export MODAL_TOKEN_ID=… MODAL_TOKEN_SECRET=…
   # optional: cp secrets.env.example → secrets.env and fill
   ./training/run_heldout_plus_modal_eval.sh
   # defaults: RUN_TAG=m9a-local-20260927-014429 MODULE=latest
   ```

6. Commit and push artifacts (explicit paths):

   ```bash
   git add reports/eval_m9a-local-20260927-014429-heldout-plus.json \
     reports/TEST-EVAL/TEST-EVAL-REPORT-heldout-plus-m9a-local-20260927-014429.md \
     reports/TEST-EVAL/TEST-EVAL-COMPARE-323-vs-1323-m9a-local-20260927-014429.md \
     reports/TEST-EVAL/eval_m9a-local-20260927-014429-heldout-plus.json \
     reports/TEST-EVAL/MANIFEST.json
   git commit -m "#112 heldout-plus: Modal L4 eval n=1323 + TEST-EVAL 323 vs 1323"
   git push -u origin <your-branch>
   ```

7. Open/update PR to `main`; close dedicated pickup issue with commit SHAs and Modal wall/cost from eval JSON / experiment record.

---

## Path B — GitHub Actions (operator, signed in)

1. Ensure repo secrets: **`MODAL_TOKEN_ID`**, **`MODAL_TOKEN_SECRET`**; optional **`HF_TOKEN`**.
2. Actions → **heldout-plus-modal-eval** → **Run workflow**  
   - **Branch:** `cursor/heldout-plus-modal-hf-optional-dfc5` (includes `push_results` + HF-optional runner)  
   - **run_tag:** `m9a-local-20260927-014429`  
   - **module:** `latest`  
   - **skip_deploy:** `false` (unless image already has current plus bake)  
   - **push_results:** `true`
3. On success: fetch branch **`heldout-plus-eval/m9a-local-20260927-014429`** or download artifact `heldout-plus-eval-m9a-local-20260927-014429`.
4. Next agent: merge results into `main` via PR; verify table in “Done when” above.

Workflow file: `.github/workflows/heldout-plus-modal-eval.yml`

---

## Path C — Pick up results only (GPU already run elsewhere)

If an operator already ran the script or GHA:

```bash
git fetch origin heldout-plus-eval/m9a-local-20260927-014429
git checkout origin/heldout-plus-eval/m9a-local-20260927-014429 -- reports/
# verify n_docs=1323, then merge to main branch via PR
```

Or:

```bash
gh run list --workflow=heldout-plus-modal-eval --limit 1
gh run download <RUN_ID> -n heldout-plus-eval-m9a-local-20260927-014429
```

---

## Commands reference

| Action | Command |
|--------|---------|
| Full one-shot (local shell with Modal) | `./training/run_heldout_plus_modal_eval.sh` |
| Interpret only | `uv run python training/interpret_heldout_plus_eval.py --eval-json reports/eval_m9a-local-20260927-014429-heldout-plus.json --baseline-json reports/eval_m9a-local-20260927-014429.json --run-tag m9a-local-20260927-014429` |
| Post-check | `jq '.eval_subset, .n_docs' reports/eval_m9a-local-20260927-014429-heldout-plus.json` |

---

## PR / merge notes

- **PR #39:** Modal-only secrets in runner + GHA `push_results` branch. CI may fail on unrelated `training/pretty_log.py` ruff — orthogonal to heldout-plus workflow.
- Merge #39 before relying on `main` workflow defaults for `push_results`.

---

## Out of scope for this pickup

- Hub publish `heldout-plus-v1` dataset (phase 2).
- Corporate top-up / CUAD mapping (see `reports/HOLDOUT-PLUS-V1-REPORT.md`).

---

## Evidence trail (agent session)

- Pool built and preflight green on snapshot VM.
- HF public download of Arm B checkpoint smoke-tested (1 doc CPU) — pipeline loads.
- No `reports/eval_*-heldout-plus.json` committed anywhere on origin through handoff commit.
