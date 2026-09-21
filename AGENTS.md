# AGENTS.md — mailroom-ml

The dedicated ML training / evaluation / fine-tuning environment for the
Digital Mailroom: the **ModernBERT ingest fast-path classifier** (one shared
encoder + per-class conditional heads), its data lineage, calibration, and
routing contract.

**This is a standalone contractor repo — NOT a governed member of the
constellation.** It serves the governed surfaces (`eval-environment`,
`llm-mailroom`, `Mailroom-Corpus-EDA`) and the `mailroom-issues` epic **#85**
intake-overhaul track. Consequences:

- Its board is `governance/TASKS.md` (light tracking), **not** the monorepo
  `DMR-00N` Kanban and **not** a package `MESSAGE_BOARD.md`.
- It **consumes** the constellation taxonomy; it may never redefine it
  (§ Taxonomy law below).
- Cross-repo coordination happens through GitHub issues in
  `LLM-Mailroom-Services/mailroom-issues` (`gh` authed as `Exios66`), not
  through the monorepo release train.

> **Current mission:** `#112` (M9a subclass-head improvement pass). Before
> touching training/eval, read `governance/M9a-HANDOFF.md` **first** — it
> carries the live diagnosis, decisions, commit trail, and the pre-registered
> Modal run config. Note that a Modal run costs real money and is gated on
> operator budget authorization.

---

## Architecture at a glance

- **One shared encoder, six conditional heads:** a `doc_type` head over 5
  classes plus one subclass head per doc_type (`contract` 26, `merger_agreement`
  5, `corporate_record` 10, `correspondence` 8, `insurance_claim` 6 —
  subclass vocabularies are **derived from observed GT at build time**, never
  hard-coded; see `mailroom_ml.labels.observed_label_surfaces`).
- **Windowing:** 8,192-token native context, 512-token overlap, plurality-vote
  merge across windows (`mailroom_ml.windows`).
- **Calibration:** per-head temperature scaling; the eval emits per-head ECE
  (`mailroom_ml.calibration`).
- **Routing:** composite score `S = p_calibrated · agreement · margin` against
  the fast-path gate; the LLM sorter remains the authority for the messy /
  ambiguous / sparse tail (`mailroom_ml.routing`, thresholds in `config.py`).
- **Serving:** ONNX int8 on CPU is the primary path
  (`artifacts/onnx/model/model_quantized.onnx`, ~60 MB, µs–ms/doc). The Modal
  serve app is an explicit **FALLBACK**.

## Layout

- `src/mailroom_ml/` — the library (imports as `mailroom_ml`)
  - `config.py` — **the interlock**: all Hub pins, taxonomy tuples, routing
    thresholds, the #85 intake contract, enrichment source pins, synthetic
    program caps. Read it fresh every session; do not hard-code its values
    elsewhere.
  - `labels.py` — canonical label surfaces + `normalize_subclass` +
    `observed_label_surfaces` (the head-vocab source), `SUBCLASS_BY_CLASS`.
  - `normalize.py` / `preprocessing.py` — `deterministic_normalize` (the
    clerk-normalization the pipeline feeds the classifier).
  - `dataset.py` / `windows.py` / `cohorts.py` — dataset assembly + windowing.
  - `inference.py` / `calibration.py` / `routing.py` — serving, temperature
    scaling, the gate.
  - `enrichment.py` / `provenance.py` / `tracing.py` — augmentation sources,
    lineage, trace sidecars.
- `training/` — the runnable drivers (not imported by the library)
  - `build_dataset.py` — corpus → staged training set.
  - `assemble_enrichment.py` — the measured augmentation ladder (authentic →
    source-matched enrichment → distillation → label-card synthesis).
  - `train_modernbert.py` — the trainer (see its CLI defaults below).
  - `eval_modernbert.py` — the eval harness (GPU-side; emits per-head metrics).
- `deploy/` — Modal layer + ONNX export. **Runbook: `deploy/README.md`.**
- `tests/` — the suite (markers: `train`, `serve`, `fullcorpus`).
- `data/modernbert_training/stage/` — the staged local training set
  (`labels.json` = the trained head maps; `vocabularies.json` = canonical
  surfaces; `manifest.txt`). `data/` is gitignored/regenerated.
- `artifacts/` — `pytorch/model/` (encoder + `heads.pt` + `labels.json` +
  `summary.json` + `temperatures.json` + `train_counts.json`), `onnx/model/`
  (serving bundle), `run2-published/` (the published snapshot).
- `reports/` — run reports + eval JSONs (e.g. `RUN3-REPORT-20260921.md`,
  `eval_run3_20260921.json`).
- `docs/` — `intake-classifier-combined-plan.md` (the approved-for-build plan)
  + `plan-amendment.md`.
- `governance/` — `TASKS.md` (board) + `M9a-HANDOFF.md` (mission state).

## Commands

```bash
# environment (uv-managed; .venv is the local venv)
uv sync --extra dev           # core + pytest/ruff
uv sync --extra train         # + torch/transformers/accelerate
uv sync --extra serve         # + onnxruntime/optimum/fastapi
uv sync --extra deploy        # + modal (deploy-time only)

# tests — surgical by default
uv run pytest -m "not fullcorpus"          # core suite, no torch/onnx needed
uv run pytest tests/test_train.py -v       # the trainer/selection seam
uv run pytest -m serve tests/test_deploy.py -v   # ONNX parity (self-skips w/o artifacts)
uv run ruff check <changed files>

# training / eval (local, only if you have a GPU)
uv run python training/train_modernbert.py --help
uv run python training/eval_modernbert.py --checkpoint artifacts/pytorch/model --json

# Modal (see deploy/README.md for the full runbook + cost notes)
HF_TOKEN=... uv run --extra deploy modal deploy deploy/modal_app.py
HF_TOKEN=... uv run --extra deploy modal run deploy/modal_app.py --epochs 5 --push-to-hub ...
#   smoke:  modal run deploy/modal_app.py --epochs 1 --limit 64
```

The core suite is green with **no** modal/torch/onnxruntime installed — every
deploy-layer test is `importorskip`/`skipif` guarded and never touches the
network. Keep it that way: new deploy tests must guard, not hard-require the
heavy extras.

## Data & Hub pin law

- **Every source is revision-pinned in `config.py`; never pull live tips.**
  The trainer's Modal app re-verifies the pinned training-data revision
  against the Hub **before** the trainer starts — a broken pin fails before
  any GPU minute.
- Canonical corpus: `Lucius-Morningstar/mailroom-dataset` @ `46a4d3c2…`
  (immutable eval corpus). Working duplicate for ML:
  `Lucius-Morningstar/mailroom-finetune` @ `19720ceb…` (the "mailroom-train"
  duplicate). Prepared training set:
  `Lucius-Morningstar/mailroom-modernbert-training` @ `5b72a345…`.
- **The current training-data revision is leak-free and clerk-normalized by
  design.** The prior 2026-09-19 build leaked the label through the title
  fallback and consumed RAW corpus text; `5b72a345` is semantic-only-titled
  and `deterministic_normalize`d so training input is byte-representative of
  inference input. Do not reintroduce a title/filename label leak, and do not
  train on raw text.
- Publishing is **operator-set per run** (`--push-to-hub` /
  `CLASSIFIER_MODEL_REPO`); never publish implicitly from a default code path.

## Taxonomy law (hard boundary)

Canonical subclass surfaces are sanctioned by `mailroom-issues` **#66/#67/#68**
+ the `llm-dojo-scoring` dojo taxonomy (`DMR-066`). **mailroom-ml consumes the
taxonomy; it may not unilaterally redefine it.** Any real dictionary change
routes **upstream as an RFC / child issue on the #85 epic** — that is a human
call, surfaced as a `needs_attention` card, never guessed around. Head label
maps are *derived from observed GT at build time*, not hand-maintained; a
surface-drift parity test (`tests/test_surfaces.py`, #66/#67/#75) guards this.
`SUBCLASS_PROJECTIONS` / `SUBCLASS_UNMAPPED_ROUTE` in `config.py` encode the
`#107` conformance projections (e.g. corporate_record's canonical
`certificate_of_formation` projects to the head's `other`; insurance has no
`other` head, so out-of-vocab subclasses route to the LLM, never silently
remapped).

## Governance & task board

`governance/TASKS.md` is the board; read it first every session. This is a
**light-tracking** board (a standalone contractor repo), so it does not carry
the monorepo's full five-lane DMR machinery — but the principles hold:

- **Claim before edit**; label the card the moment a diff exists.
- **Discover → spawn:** anything found but not delivered becomes its own
  `mailroom-issues` issue *before* the parent card closes (#113–#116 were
  spawned this way from #112).
- **Close with proof:** green suites for the touched scope, clean `git status`
  for the card's scope, Evidence naming the commit(s).
- **Commit discipline:** prefix with the card it affects (`#NNN …` — this
  repo's convention; it has no `DMR-` prefix of its own), detailed bodies
  naming files, targeted staging (`git add <explicit paths>`, never
  `git add .`), UTC timestamps.

Cross-cutting work is tracked in `mailroom-issues`; this board mirrors the
local mission state. Do not commit another session's in-flight files
(`git status --porcelain` before every commit).

## Specialist subagents — call them, don't impersonate them

Invoke a specialty through the **Task tool** (`subagent_type: <name>`) rather
than improvising outside your expertise; specialty work done inline is a
defect even when it happens to be correct. The caller owns the deliverable —
a subagent's report is evidence, not a merged change; **you** write the files,
run the gates, update the board, and commit.

Roster most relevant to this repo (see the monorepo `Digital-Mailroom/AGENTS.md`
for the full constellation roster):

| Specialty | `subagent_type` | Call it for |
|---|---|---|
| HF & data science | `lucius` (project) | Hub downloads/uploads, `datasets`/`transformers` pipelines, EDA, statistical analysis, training/eval |
| Data & databases | `athena-database-agent` | dataset selection/integration, schema, ingestion, data QA, SQL |
| Modal compute | `modal-specialist` (project) | Modal apps/images/volumes/secrets/GPU/cost, cold starts — **verify the current SDK docs first** |
| Prompt engineering | `prompt-engineer` (project) | run-failure diagnosis, GEPA mutations, prompt A/Bs, plateau calls |
| Board & repo auditing | `board-evidence-auditor` (project) | verify card claims vs shipped evidence, TODO sweeps, STATE.md trail |
| Docs & board | `atom` | READMEs/changelogs, doc-drift, board upkeep |
| Code analysis | `code-analyst` | fast verdicts, diff review, root-cause tracing, dead-code confirmation |
| Test suite auditing | `test-suite-auditor` | whether the suite would catch a regression; stub honesty; gap lists |
| Systems & hygiene | `jarvis-systems-maximizer` | performance, memory/disk, repo bloat |
| File organization | `archivist-file-organizer` | layout audits/restructures, junk/duplicate sweeps |
| Codebase exploration | `explore` | fast read-only location ("where is X") before you edit |
| Mission execution | `general` | multi-step work with no single owner |

**Roster liveness caveat (this repo).** The project specialists
(`lucius`, `prompt-engineer`, `board-evidence-auditor`) live in
`.opencode/agents/` and are mirrored from the monorepo. opencode fixes its
subagent roster **at session start**: adding or editing an agent file (or
compacting the context) does **not** make it callable — only a genuine
opencode restart does (verify with `opencode agent list`). If a needed
specialist is not in the Task tool's roster this session, do not fabricate the
dispatch: run the work under `general` with that specialist's protocol as the
brief, and note the substitution in the card Evidence. `PROMPT_ENGINEER_GEPA_PROVENANCE.md`
is provenance documentation, **not** a callable subagent.

Rules: **one specialist per concern** (chain them in dependency order — e.g.
`explore` → `modal-specialist` → `lucius`); **brief like a card** (card ID,
exact scope, seams, evidence contract); **specialists verify current upstream
docs** before writing configuration.

## Deploy (Modal) & cost discipline

- **Modal is deploy-time only** — never part of the runtime venv; tests stub
  or `importorskip` it. The SDK is pinned in `pyproject.toml` (`modal==1.5.5`,
  verified 2026-09-18).
- Training app `mailroom-ml-train` (`deploy/modal_app.py`), serving fallback
  `mailroom-ml-serve` (`deploy/serve_app.py`). Checkpoints live on the
  `modernbert-checkpoints` Volume: `/checkpoints/latest/` (the consumed
  pointer) and `/checkpoints/runs/<run-id>/` (per-run archive, rollback).
  Failed runs never commit — `latest/` keeps the last good checkpoint.
- **Cost is real.** A full run is a multi-hour L4 job (order of dollars); the
  plan budget is ~1–2 L4-hours one-time. A smoke run is minutes. `modal-specialist`
  will refuse to `--force` past a stated money ceiling — **budget overrides are
  an operator decision**, surfaced as a `needs_attention` card, never a silent
  `--force`. Measure s/step before quoting an estimate; do not trust a
  remembered figure.
- ONNX export + parity gate: `deploy/onnx_export.py` →
  `deploy/onnx_parity_check.py` (fp32 ≤1e-4 is the correctness contract; int8
  gated on argmax agreement). **Do not export through `optimum-cli`** — the
  shared-encoder + `heads.pt` architecture would initialize random heads and
  silently ship a broken graph; use the committed `torch.onnx.export` path
  (`dynamo=False`).

## Testing & documentation currency

- Markers are declared in `pyproject.toml`; run the surgically relevant suite
  by default and the full non-`fullcorpus` suite for cross-cutting changes.
- **A behavior change ships its tests and its docs in the SAME commit** — if a
  change alters what `README.md`, `AGENTS.md`, or `governance/TASKS.md`
  describes, update those files together with the code.
- Selection/gate behavior must stay **pinned**: the checkpoint-selection rule
  (`_select_epoch`, the doc_type gate + ECE budget + subclass objective) has
  tests that fail if reverted. If you change selection, the eval's `per_head`
  output, or the loss weighting, update the pinning tests in
  `tests/test_train.py` / `tests/test_eval_cli.py` in the same commit.
- Do not commit heavy assets to the repo root (screenshots, panels); reference
  or relocate them. If you find untracked scratch files you did not create,
  leave them and note them — do not delete another session's work.