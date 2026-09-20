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
