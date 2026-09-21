# M9a — Mission Handoff / State (mailroom-issues #112)

**Written: 2026-09-21 (UTC). Author: orchestrator. Purpose: post-compaction
resume with full state. Read this + `governance/TASKS.md` before acting.**
**Do NOT launch the Modal run until the operator authorizes the budget.**

---

## 1. Mission

- **Card:** mailroom-issues **#112** — "#85 M9a follow-up: contract +
  correspondence subclass heads — next improvement pass (run-3 gap)".
- **Repo:** `mailroom-ml` (standalone contractor repo; board is
  `governance/TASKS.md`, not a constellation-governed member).
- **Parent epic:** #85 (ModernBERT-coupled intake overhaul). Related: #107
  (run-3 close), #66/#67/#68/#75 (taxonomy surfaces).
- **Success gate (from #112):** contract test macro-F1 **≥ 0.20** AND
  correspondence **≥ 0.25**, with doc_type test acc **≥ 0.89** and window
  ECE **≤ 0.05**, on the held-out test via the GPU eval harness.
- **State:** `status/in-progress`; claimed by orchestrator 2026-09-21.

## 2. Diagnosis (evidence-backed — DO NOT re-litigate)

**The subclass heads suffer majority-prior collapse, not capacity failure.**

| head | classes | val window-acc | val macro-F1 (obs) | reading |
|---|---|---|---|---|
| insurance_claim | 6 (near-balanced) | 0.8687 | 0.7677 | **learned — the control** |
| corporate_record | 10 | 0.5000 | 0.2156 | partial collapse |
| merger_agreement | 5 | 0.5267 | 0.1951 | partial collapse |
| correspondence | 8 | 0.5106 | 0.0845 | collapse to `email` (val share 0.511) |
| contract | 26 | 0.1304 | 0.0395 | ≈ majority-prior |

Key results (agents: `code-analyst` U1, `athena-database-agent` U2,
`general` under lucius protocol U3):

1. **Same head, same representation** — the only near-balanced head learns;
   every imbalanced head collapses. Capacity is excluded.
2. **Heads sit exactly on the prior** — correspondence window-acc 0.5106 ==
   val email share 0.511; merger 0.5267 ≈ `other` share 0.534. The correct
   baseline is the **majority share**, NOT 1/K.
3. **The signal is present** — a bag-of-words NB reaches **contract macro-F1
   0.529 / correspondence 0.485** (grouped 5-fold CV) vs the model's
   0.0395/0.0845. Data-signal scarcity is falsified.
4. **Dictionaries are scope-complete** — 0.00% true `other`-fallthrough,
   0 drift keys, staged `subclass` == `normalize_subclass(raw)` for 100% of
   rows; the contract 8-char prefix matcher (`labels.py:181-184`)
   mis-assigns nothing (dead code for this corpus). **The dictionary-scope
   hypothesis is falsified** for the pinned corpus.
5. **Natural experiment pre-supports the fix** — run-2 used the trainer
   default `--weight-mode inverse`; run-3 switched to `sqrt-inverse` and got
   *worse* tail heads (0.101→0.0395 / 0.098→0.0845) while gaining on
   doc_type/insurance. Restoring full inverse-frequency weighting is the
   best-supported single change.
6. **Metric caveat** — contract val has 21 observed classes: 5 zero-support,
   6 n==1, 19/21 n<5. A 26-class macro-F1 is **not statistically
   interpretable** on this split; always read it beside per-class support
   (the eval now emits `per_head.<head>.support`).

Contributing mechanics (file:line):
- `train_modernbert.py:258` — `loss = λ·CE_dt + (1-λ)·mean(sc_ces)`; λ=0.65 →
  each subclass head gets ≈0.09 of the backbone gradient (U1).
- `train_modernbert.py:213-227` + `labels.json` — `sqrt-inverse` + cap 10
  leaves the majority class the top effective mass.
- `train_modernbert.py:250-256` — each head trains only on its own rows, hard
  targets; label-smoothing is applied to `dt_ce` ONLY (so it is a **no-op**
  for the failing heads).

## 3. Decisions made

- **Hypothesis chosen:** loss-side first — restore/strengthen imbalance
  weighting (+ gradient share), because (4) and (3) above falsify the
  data/dictionary alternatives as the first lever.
- **Rejected for this pass:** (i) dropout/label-smoothing sweep — collapse ≠
  overfitting and smoothing isn't wired to subclass CE; (ii) λ_dt trade
  ALONE — doc_type acc 0.8947 vs 0.89 floor, margin too thin (though the new
  selection gate now bounds this risk).
- **Prerequisites fixed before any run** (so the gate is measurable and
  subclass progress isn't discarded).
- **Taxonomy governance:** canonical subclass surfaces are sanctioned by
  #66/#67/#68 + the dojo taxonomy source. mailroom-ml **consumes** them and
  may NOT unilaterally redefine them. Any real dictionary change routes
  upstream as an RFC/child issue on #85 — a human call.

## 4. Commits (repo `mailroom-ml`, branch `main`)

| commit | scope |
|---|---|
| `1718358` | U4 — eval emits `per_head.{head}.{macro_f1,support}`; checkpoint selection is lexicographic (`DOC_TYPE_GATE_TOL=0.005` + `ECE_BUDGET=0.05`, then maximise mean subclass macro-F1); `--select-on-subclass` default on. Verified: selects run-3 epoch 2 (`subclass_objective` 0.3003). |
| `1081a6e` | U7b — extracted pure `_select_epoch(...)` seam; +10 tests pinning the gate/flag/`per_head`; revert now FAILS a test. 233 passed / 3 skipped. |
| `18b8a4f` | board claim + deployed `lucius`, `prompt-engineer`, `board-evidence-auditor` into `.opencode/agents/`. |
| `4e1280a` | board ledger (wave 1–3 results). |

## 5. Spawned issues (filed 2026-09-21)

- **#113** CUAD contract corpus pinned but not wired into `enrichment.py`
  (509 rows, all 25 keys) — `priority/high`.
- **#114** `assemble_enron_gt` not subclass-balanced — Tier-1 adds ~824
  ~98%-email rows; **would deepen the collapse. Do NOT run
  `assemble_enrichment.py --tiers 1` for correspondence before this is
  fixed** — `priority/high`.
- **#115** `mixture_caps` greedy/sorted-name allocator starves
  late-alphabet subclasses — `priority/medium`.
- **#116** contract head zero-train/zero-val `other` class (26-class head,
  25 weights) — `priority/medium`; needs a tracker decision (a) keep as
  inference-only fallback, or (b) drop it.

## 6. Pending action — the Modal run (BLOCKED on budget)

**Operator has chosen: ONE arm, 3 epochs** (pending final arm pick & budget
authorization).

**Pre-registered arms** (run-3 config except as noted; `--select-on-subclass`
default ON):
- **Arm B (recommended single arm):** `--weight-mode inverse --weight-cap 20
  --loss-lambda-dt 0.65` — pure weighting isolate; holds doc_type gradient
  share constant → zero risk to the locked doc_type gate. Tests the primary
  pre-supported hypothesis.
- **Arm A (aggressive alt):** same + `--loss-lambda-dt 0.5` — adds the
  subclass gradient-share lever; highest gate probability, but confounded
  (2 levers) and spends the doc_type-risk lever (bounded by the new gate).

**Cost (smoke-measured, 6.0 s/step, 1125 steps/epoch):**
- 3 epochs ≈ **$6.0**; 4 epochs ≈ $7.79 per arm. Guard ceiling is **$4.32**
  → a 3-epoch run needs `--budget 7` (or explicit `--force`). Operator is
  cost-sensitive; 3 epochs is the chosen envelope.

**Exact spawn (Arm B shown; `HF_TOKEN` from
`~/.config/opencode/secrets/hf-token`):**
```bash
cd /Users/morningstar/Desktop/Cold_Storage/mailroom-ml
python deploy/modal_app.py   # (already redeployed at commit 1718358; redeploy
                             #  again if the tree changed)
python deploy/spawn_train.py --epochs 3 --batch-size 4 --grad-accum 8 \
  --lr 2e-5 --seed 42 --budget 7 \
  --trainer-extra=--mlp-heads --trainer-extra=--label-smoothing=0.05 \
  --trainer-extra=--max-length=8192 --trainer-extra=--weight-decay=0.01 \
  --trainer-extra=--warmup-frac=0.06 --trainer-extra=--early-stop-patience=2 \
  --trainer-extra=--weight-mode=inverse --trainer-extra=--weight-cap=20 \
  --trainer-extra=--loss-lambda-dt=0.65
```
- **App:** `ap-KVX4EVKG4MI2XJ71r6Hkow`; arms are **serial** (double-launch
  guard); no arm-name label — distinguish by `run_id` + the echoed flags.
- **Poll:** `modal app logs --follow ap-KVX4EVKG4MI2XJ71r6Hkow`; archives at
  `/checkpoints/runs/<run_id>`.
- **Eval:** run `training/eval_modernbert.py` (GPU harness; see
  `deploy/eval_app.py`) against the selected checkpoint and read
  `per_head.contract.macro_f1` / `per_head.correspondence.macro_f1`
  (+ `support`).
- **Do NOT push to Hub** until the caller decides on publish.

**Modal SDK:** 1.5.5 (verified latest stable 2026-09-21; `pyproject.toml`
deploy extra pins `modal==1.5.5`).

## 7. Ops seams / open items

- **Agent roster gap:** `lucius`, `prompt-engineer`, `board-evidence-auditor`
  were deployed to `.opencode/agents/` (commit `18b8a4f`) but opencode fixes
  its roster at session start — they are **live next session**. This session
  used `general` under the lucius protocol for U3. Use the real agents
  post-restart.
- **Pending cards:** U5 (dictionary/definitional scope — label cards +
  contrastive synthesis; upstream RFC gate, `prompt-engineer`), U8 (board
  evidence audit, `board-evidence-auditor`), U9 (docs/board close, `atom`).
- **Untracked junk** (not ours): `archive-browser-b4-evidence.png`,
  `bert-panels.png`.
- **Full-arm question (for the operator):** both arms buys the *interaction*
  (marginal effect of λ_dt given weighting restored) at ~2× cost and spends
  the doc_type-risk lever. One arm (B) is cheaper, isolates the primary
  hypothesis, and — with the new lexicographic gate — is now judged on its
  best subclass epoch with doc_type protected. **Sequential > parallel here.**

## 8. Resume checklist

1. Read `governance/TASKS.md` + this file.
2. Confirm the agent roster now includes `lucius`/`prompt-engineer`.
3. **(Operator)** authorize budget + arm pick → launch §6.
4. On completion: eval → compare against the §1 gate → U8 audit → U9 close.