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
| U3b label-surface derivation + parity tests (#66/#67/#75) | athena | ✅ done | `labels.py` observed_label_surfaces + `tests/test_surfaces.py`; commits `2b7e120`/`631c5e5` |
| U4 deploy layer (Modal, current SDK) | modal-specialist | ✅ done | deploy/ + runbook; SDK 1.5.5 verified |
| U5 model/inference/calibration/routing/training + enrichment | lucius (linked session) | ✅ done | `src/mailroom_ml/inference.py` ModelBundle + training drivers; M9a handoff commit trail |
| U6 integration + full suite | orchestrator | ✅ done | `uv run pytest -m "not fullcorpus"` green on main |
| U7 commit + report | orchestrator | ✅ done | `reports/RUN3-REPORT-20260921.md` + eval JSON |

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
email share 0.57 train / 0.511 val, macro-F1 0.0845 ≈ the collapse floor;
`contract` window-acc 0.13 vs chance 1/26 = 0.038, macro-F1 0.0395 ≈ chance).

| Card | Owner | Status | Evidence |
| --- | --- | --- | --- |
| M9a-U0 mission plan + board claim | orchestrator | ✅ done | this card; issue #112 comment |
| M9a-U1 root-cause verdict (loss seam) | code-analyst | ✅ done | majority-prior collapse confirmed; insurance=head control; selection gate was doc_type-only |
| M9a-U2 data/taxonomy QA (normalization, support floor) | athena | ✅ done | **dictionaries scope-complete + drift-free (0.00% fallthrough, 0 drift keys); prefix matcher safe** |
| M9a-U3 corpus EDA + separability + augmentation eligibility | general (lucius protocol) | ✅ done | **lexical NB: contract macro-F1 0.529 / correspondence 0.485 vs model 0.0395/0.0845 → loss-side first** |
| M9a-U4 eval-harness gap + selection gate | general | ✅ done | commit `1718358`; selection rule would pick epoch 2 on run-3 val — **not persisted** in shipped `summary.json` (epoch 1 legacy pick) |
| M9a-U7 test audit | test-suite-auditor | ✅ done | commit 1718358 was 100% unpinned; gap list (10) delivered |
| M9a-U7b pin selection + per_head | general | ✅ done | commit `1081a6e`; 233 passed / 3 skipped; revert now fails a test |
| M9a-U5 dictionary/definitional scope (label cards, contrastive synthesis) | prompt-engineer | ⏸ deferred (post-run) | U2 falsified dictionary-scope for the pinned corpus (0 fallthrough); only a contingent data lever if the loss-side run under-delivers. Any real dictionary change routes upstream as an RFC on #85 — human call |
| M9a-U6 L4 run (operator chose ONE arm, 3 epochs) | modal-specialist + lucius | ⛔ **BLOCKED — budget** | smoke GREEN; 3-epoch arm ≈ $6.0 vs $4.32 guard; **no spend**; a restarted-session agent executes §6 |
| M9a-U8 board evidence audit | general (board-evidence-auditor protocol — not callable pre-restart) | ✅ done | commit `f6a49a1` + audit: all execution claims VERIFIED (diffs, 233/3/3, issues #113–#116); 4 forward-narrative corrections landed |
| M9a-U9 docs/board close | atom | ⏳ pending | — |

### U6 run configuration (decided: ONE arm, 3 epochs — smoke-validated)

**Operator decision:** a single arm, 3 epochs (not the earlier two-arm/4-epoch
draft). Arm **B** — the pure weighting isolate — is the recommendation:
run-3 config except `--weight-mode inverse --weight-cap 20
--loss-lambda-dt 0.65`, `--epochs 3`, `--select-on-subclass` (default on).
Arm **A** (alt, same + `--loss-lambda-dt 0.5`) adds the subclass
gradient-share lever at ~2× reflex risk; keep it serial/second.
App `ap-KVX4EVKG4MI2XJ71r6Hkow` redeployed from the working tree (commit
1718358 confirmed live via the smoke's lexicographic-rule line).

**`needs_attention` — M9a-U6 budget override (human call).** Smoke-measured
6.0 s/step, 1125 steps/epoch → one 3-epoch arm ≈ **$6.0** all-in vs the guard
ceiling `$4.32`. Launch needs `--budget 7` (or explicit `--force`) — an
operator-authorized override; `modal-specialist` correctly refuses to
`--force` past a stated ceiling silently. Run-3 was 2 epochs / $4.11.
**A restarted-session agent with the full specialist roster executes this
section**; the exact command is in `governance/M9a-HANDOFF.md` §6.

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
