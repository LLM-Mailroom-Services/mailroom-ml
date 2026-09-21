# mailroom-ml — Mission Board (standalone contractor repo)

Light tracking only. This repo is NOT a governed member of the constellation;
it serves Digital-Mailroom's governed repos (eval-environment, llm-mailroom,
Mailroom-Corpus-EDA) and the mailroom-issues epic #85 track. The board exists
so concurrent sessions (this one + the linked lucius session, share
ih12eEK0) do not collide.

## Mission: ModernBERT ingest fast-path classifier (combined plan)

Plan: `docs/intake-classifier-combined-plan.md` (v2026-09-18, issues-minned).
Corpus: `mailroom-finetune @ 19720ceb…` (modifiable duplicate; canonical
`mailroom-dataset @ 46a4d3c2…` untouched). Model: `answerdotai/ModernBERT-base`.

| Card | Owner | Status | Evidence |
| --- | --- | --- | --- |
| U0 recon + fact verification | orchestrator | ✅ done | §2 fact base, hub-verified 2026-09-18 |
| U1 skeleton + config interlock | orchestrator | ✅ done | config.py all pins/constants |
| U2 combined plan doc | orchestrator | ✅ done | docs/intake-classifier-combined-plan.md |
| U2b issues incorporation (#84–#92, #66–#68, #52, #57, #75) | orchestrator | ✅ done | plan §15 matrix; config.py BERT_INTAKE_*, surface canons |
| U3 data layer port (mailroom-finetune) | athena | ✅ done | 28+9 tests; byte-compat statement; commit 703352d |
| U3b label-surface derivation + parity tests (#66/#67/#75) | athena | 🔄 in flight | observed_label_surfaces + fullcorpus parity |
| U4 deploy layer (Modal, current SDK) | modal-specialist | ✅ done | deploy/ + runbook; SDK 1.5.5 verified |
| U5 model/inference/calibration/routing/training + enrichment | lucius (linked session) | 🔄 in flight | model.py…; share ih12eEK0 |
| U6 integration + full suite | orchestrator | ⏳ pending | uv run pytest -m "not fullcorpus" |
| U7 commit + report | orchestrator | ⏳ pending | this board |

## Mission M9a — subclass-head improvement pass (mailroom-issues #112)

> **Post-compaction resume: read `governance/M9a-HANDOFF.md` FIRST** — it
> carries the full diagnosis, decisions, commits, run config, and the
> exact spawn command. Do not launch the Modal run before the operator
> authorizes the budget.

Spawned from #107 close. Goal gate: **contract macro-F1 ≥ 0.20 AND
correspondence ≥ 0.25 on the held-out test via the GPU eval harness, with
doc_type accuracy ≥ 0.89 and window ECE ≤ 0.05.**

Claimed by: orchestrator (primary session), 2026-09-21.

Diagnosis (evidence-backed, this session): the subclass heads exhibit
**majority-class collapse**, not capacity failure. `insurance_claim` — the
only near-balanced head (6 classes) — reaches macro-F1 0.77 / window-acc
0.87; every imbalanced head collapses (`correspondence` window-acc 0.51 ≈
email share 0.57, macro-F1 0.0845 ≈ the collapse floor; `contract`
window-acc 0.13 vs chance 0.038, macro-F1 0.0395 ≈ chance).

| Card | Owner | Status | Evidence |
| --- | --- | --- | --- |
| M9a-U0 mission plan + board claim | orchestrator | ✅ done | this card; issue #112 comment |
| M9a-U1 root-cause verdict (loss seam) | code-analyst | ✅ done | majority-prior collapse confirmed; insurance=head control; selection gate was doc_type-only |
| M9a-U2 data/taxonomy QA (normalization, support floor) | athena | ✅ done | **dictionaries scope-complete + drift-free (0.00% fallthrough, 0 drift keys); prefix matcher safe** |
| M9a-U3 corpus EDA + separability + augmentation eligibility | general (lucius protocol) | ✅ done | **lexical NB: contract macro-F1 0.529 / correspondence 0.485 vs model 0.0395/0.0845 → loss-side first** |
| M9a-U4 eval-harness gap + selection gate | general | ✅ done | commit `1718358`; run-3 archive now selects epoch 2 |
| M9a-U7 test audit | test-suite-auditor | ✅ done | commit 1718358 was 100% unpinned; gap list (10) delivered |
| M9a-U7b pin selection + per_head | general | ✅ done | commit `1081a6e`; 233 passed / 3 skipped; revert now fails a test |
| M9a-U5 dictionary/definitional scope (label cards, contrastive synthesis) | prompt-engineer | ⏳ pending | upstream RFC gate; CUAD unwired, Enron Tier-1 unbalanced |
| M9a-U6 two-arm L4 run | modal-specialist + lucius | ⛔ **BLOCKED — budget** | smoke GREEN; est. $7.79/arm vs $4.32 ceiling; **no spend** |
| M9a-U8 board evidence audit | board-evidence-auditor | ⏳ pending | — |
| M9a-U9 docs/board close | atom | ⏳ pending | — |

### U6 run configuration (pre-registered, smoke-validated)

Arms: run-3 config except — **A**: `--weight-mode inverse --weight-cap 20
--loss-lambda-dt 0.5`; **B**: `--weight-mode inverse --weight-cap 20
--loss-lambda-dt 0.65`. Both `--epochs 4`, `--select-on-subclass` (default
on). App `ap-KVX4EVKG4MI2XJ71r6Hkow` redeployed from the working tree
(commit 1718358 confirmed live via the smoke's lexicographic-rule line).

**`needs_attention` — M9a-U6 budget decision (human call).** Smoke-measured
6.0 s/step → **$7.79 per 4-epoch arm; both arms ≈ $15.58**, vs the guard
ceiling `$4.32` (~4.4 L4-h) and the plan's stated 1–2 L4-h one-time.
Run-3 was 2 epochs / $4.11. `modal-specialist` correctly refused to `--force`
past a stated money ceiling. Options: (a) both arms 4 ep ≈ $15.6;
(b) one arm first (B — the pure weighting isolate) 4 ep ≈ $7.8; (c) both
arms 2 ep ≈ $8.2; (d) one arm 2 ep $4.11 (fits today's ceiling).
Awaiting operator call before any spend.

### Structural hazards spawned as issues (2026-09-21)

- **#113** CUAD corpus pinned but unwired (509 rows, all 25 contract keys).
- **#114** `assemble_enron_gt` not subclass-balanced — Tier-1 would deepen
  the email collapse. **Do not run `--tiers 1` for correspondence** first.
- **#115** `mixture_caps` greedy/sorted-name allocator starves late-alphabet
  subclasses.
- **#116** contract head zero-row `other` class (26-class head, 25 weights).

**Hard governance gate:** canonical subclass surfaces are sanctioned by
mailroom-issues #66/#67/#68 + the dojo taxonomy source. mailroom-ml CONSUMES
the taxonomy; it may not unilaterally redefine it. Any real dictionary change
routes upstream as an RFC/child issue on the #85 epic — `needs_attention`.

## Seams for the lucius session (shared contract)

- `mailroom_ml.config` is the interlock — read it fresh; it now includes
  BERT_INTAKE_MIN_CONFIDENCE=0.92, BERT_INTAKE_MAX_CHARS=30_000,
  GATE_REQUIRED_AGREEMENT, INTAKE_HANDOFF_SCHEMA_VERSION, enrichment source
  pins, and canonical surface tuples.
- `mailroom_ml.labels.observed_label_surfaces(docs)` (landing with U3b) is
  the head-vocab source — heads come from observed GT (correspondence 8,
  corporate_record 10, insurance 6), per issues #66/#67. Do not hard-code
  head label lists.
- Routing must honor the epic #85 contract: MAILROOM_BERT_INTAKE flag,
  BERT_INTAKE_MODE shadow|verify|skip, P1–P7/F1–F7 gate vocabulary,
  fail-open invariant.
- Do not edit: config.py, deploy/, docs/, governance/. Do not commit the
  other session's in-flight files.
