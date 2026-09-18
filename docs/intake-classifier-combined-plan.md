# Ingest Fast-Path Classifier — Combined Implementation Plan (ModernBERT)

**Status:** Approved-for-build · **Date:** 2026-09-18
**Host repo:** `mailroom-ml` (standalone contractor repo — supports the
governed constellation; not governed itself)
**Sources synthesized:**
1. `eval-environment/docs/intake-classifier-proposal.md` (2026-09-17, Ponytail Hermes Chan)
2. `ModernBERT implementation.md` (plan B — refined spec)
3. Committed work `Exios66/Mailroom-Corpus-EDA @ cf096fa` (`modernbert/`:
   `prep.py`, `publish.py`, `train.py`, `modal_app.py`, `tests/test_prep.py`)
   + published `Lucius-Morningstar/mailroom-modernbert-training`
4. Hub fact-verification pass (2026-09-18, this session) + operator direction
   (`mailroom-finetune` is the modifiable working copy)

---

## 1. Executive summary

Deploy a **fine-tuned ModernBERT-base hierarchical classifier** as a
deterministic fast-path inside the ingest node: one shared encoder, six
conditional heads (doc_type + per-class subclass), windowed long-document
coverage with the sorter's plurality-vote merge, per-head temperature
calibration, a composite route score `S = p·a·m`, and a fast-path gate that
**skips the LLM sorter only for clean, confidently-supported documents** —
validated by selective-risk analysis, not inherited thresholds. The LLM
sorter remains the calibrated authority for the messy / ambiguous / sparse
tail. Training data grows along a **measured augmentation ladder**:
authentic corpus → **source-matched enrichment** (Enron dedup ground truth,
CMS/GNOTHEIA/BDR/INSURBIAS insurance pools — the new pillar this plan adds)
→ Enron pseudo-label distillation → constrained LLM label-card synthesis.
Serving is ONNX int8 on CPU (~$0.000001/doc, µs–ms); training is a one-time
~1–2 L4-hour Modal run.

This plan is the union of the two source plans with three decisive merges:
**shared encoder + conditional heads** (plan B — already how the committed
work is built), **LLM-skip fast path gated by selective risk** (plan B,
with plan A's pipeline bands as the *initial* policy), and **source-matched
enrichment as the primary augmentation mechanism** (new — neither plan
designed for the Enron/insurance pools the operator surfaced).

---

## 2. Fact base (all verified live, 2026-09-18)

### 2.1 Model — `answerdotai/ModernBERT-base` (model card + arXiv 2412.13663)

| Claim | Value | Status |
|---|---|---|
| Params / layers | 149,655,232 / 22 layers | verified (Hub API) |
| Native context | 8,192 tokens (RoPE + local-global alternating attention) | verified |
| Pretraining | 2T tokens English + code | verified |
| GLUE (base) | 88.4 — best among similarly-sized encoders | verified (card Table 1) |
| GLUE (large) | 90.4 — second only to DeBERTa-v3-large (91.4) | verified |
| License | Apache-2.0 | verified |
| transformers | ≥ 4.48.0 | verified (card usage) |
| Notes | Unpadding + Flash Attention; no token-type IDs; no pooler — classification uses the `<s>` first-token embedding | verified (card; train.py already does this) |

### 2.2 Papers

- **arXiv 2406.08660** — "Fine-Tuned 'Small' LLMs (Still) Significantly
  Outperform Zero-Shot Generative AI Models in Text Classification"
  (Bucher & Martini, 2024): fine-tuned small encoders beat zero-shot LLM
  prompting for classification. Verified metadata + abstract.
- **arXiv 2412.13663** — ModernBERT paper. Verified.
- Not used: any unverifiable "SFRSA / Springer 2026" citation from plan A —
  replaced by the concrete label-card policy from plan B + the source-matched
  enrichment below. (Factuality rule: no citation we cannot verify survives.)

### 2.3 Corpus + data lineage (Hub API, exact revisions)

| Repo | Revision | Role | Verified facts |
|---|---|---|---|
| `Lucius-Morningstar/mailroom-dataset` | `46a4d3c240a36671cde0182fff4960f6b8b73aca` | canonical eval corpus, immutable | 3,302 rows (train 2,979 / test 323); 55 strata; 5 doc_types; `ground_truth` 36 cols incl. matter/group/relationships + `gt_fields` |
| `Lucius-Morningstar/mailroom-finetune` | `19720ceb4e29bc3134a88507aa57cdfac7a64a1b` | **modifiable working copy** (the "mailroom-train" duplicate) | created 2026-09-18; schema/rows identical to canonical (verify hashes at load — see §5) |
| `Lucius-Morningstar/mailroom-modernbert-training` | `6790341e25229a2617914c839ff0e70c590d7b6d` | prepared training set (committed build) | documents 3,302 (2,680/299/323); windows 5,068 (4,573/495); `labels.json`/`vocabularies.json`/`manifest.txt` sidecars |

### 2.4 Augmentation source pools (Hub API, 2026-09-18) — the enrichment pillar

| Pool | Repo (verified) | Content | License | Corpus already uses | Pool headroom |
|---|---|---|---|---|---|
| Enron dedup | `Lucius-Morningstar/enron-correspondence-dedup` | 247,523 unique-text emails (517,390 → dedup); `blind` = unlabeled bulk, `ground_truth` = enriched labeled subset (small) | other (card) | 1,000 correspondence rows (637 llm_zero_shot + 162 aeslc_join + 105 heuristic + 96 manual) | **~247K unlabeled** (pseudo-label + OOD/drift pools); labeled GT subset for direct expansion where labels exist |
| CMS DE-SynPUF | `Lucius-Morningstar/cms-desynpuf-insurance-claims` (rendered EOB pool, 400 rows) + raw DE-SynPUF Sample 1 (public CMS synthetic) | carrier/inpatient/outpatient/pde EOB docs | other (public-use synthetic) | 600 rows | render more via the existing renderer; effectively unbounded raw source |
| GNOTHEIA | `gratex/GNOTHEIA-synthetic-insurance-dataset` | 863 polycontext claim documents + 490 rules | Apache-2.0 | 200 property | ~660 property docs addable |
| BDR motor | `bdr-ai-org/insurance-motor-claims-decision-v1` | 800 **tabular** records (needs the same decision-letter render step the corpus used) | MIT | 150 auto | 650 records → rendered letters if renderer reproducible; else skip |
| INSURBIAS | `feihuangfh/INSURBIAS` | `Dataset/input_dataset.csv` (102.6 MB claim narratives) | CC-BY-4.0 | 150 narratives | thousands of pooled narratives (verify exact count at load) |
| CUAD/MAUD/S1 (related) | `mailroom-cuad-contracts-full`, `mailroom-maud-contracts`, `mailroom-s1-corporate-records` | contract families, merger consideration, corporate records | per-source (verify at build) | 509 CUAD, 152 MAUD, 450 S1 | same mechanism; structure to verify at enrichment build time |

---

## 3. Design decision log (which plan won, and why)

| # | Decision | Source | Rationale |
|---|---|---|---|
| D1 | One shared ModernBERT encoder + 6 heads (doc_type 5+unknown; contract 26; merger 5; corporate_record 11; correspondence 9; insurance_claim 6) | Plan B (committed work already implements) | Data-efficient vs six 149M checkpoints; matches pipeline's conditional subclass structure |
| D2 | Classifier output is a **verified fast path** that can **skip the LLM sorter** (not merely a prior) | Plan B | Without skip there is no cost reduction (plan A's own stated goal) |
| D3 | Skip gated by **selective risk on a calibration set**; pipeline bands (0.88/0.97) are the *initial* policy only | Plan B, softened | A temperature-scaled encoder probability is not automatically comparable to LLM confidence; band constants stay as deploy-time defaults, replaced once measured |
| D4 | Composite route score `S = p_calibrated × a × m` (agreement × margin) | Plan B | One window being confident is not enough; ties broken by #high-conf windows, mean conf, first-window conf |
| D5 | Training input = committed v1 format (`title + "\n\n" + window`, ≤8,192 tokens, BPE-clamped) | Committed work | Byte-compatible with the published training set; tagged `[FILE_NAME]/[TITLE]/[WINDOW_INDEX]` prefix = optional v2 config flag, not default |
| D6 | Splits: keep committed filename-level 90/10 stratified as baseline; **family-grouped split** (matter/group/thread keys) as the upgrade path the moment Enron threads enter training | Committed + Plan B | Enron threads are document families; row-level splits leak near-duplicates → misleading validation |
| D7 | Augmentation ladder: authentic-only → **source-matched enrichment** → Enron pseudo-labels → constrained LLM synthesis | **New** + Plan B | Source-matched rows are distribution-identical and label-exact (cheapest, safest); LLM synthesis only for labels with no authentic floor |
| D8 | Serving: ONNX int8 CPU primary; Modal L4 fallback; no OpenRouter | Plan A | 149M int8 ≈ 60 MB, loads in ms, zero per-call cost |
| D9 | Eval: classifier-first (correctness, calibration, selective risk, leakage) **plus** cost/accuracy parity vs the LLM sorter on the same stratified 50 | Plan A (caller's original ask) + Plan B scope note | The cost-reduction claim is only provable head-to-head vs the LLM path |
| D10 | Fail-open: any ML failure (artifact missing, load error, label-map mismatch, exception) routes to the existing LLM/deterministic path; never silently classify | Plan B | Pipeline invariant |
| D11 | Test split (323) never touches: fitting, calibration, threshold tuning, augmentation selection, or error-driven changes | Committed + Plan B | Eval surface integrity |

---

## 4. Architecture — the classification cascade

```
raw document
  │
  ▼
[intake_node]
  1. claim + transcribe                       (unchanged, deterministic)
  2. apply_intake → deterministic clerk       (unchanged, never skipped)
        └─ deterministic_normalize + looks_messy
  3. ModernBERT classifier  (NEW, in-node function)
        ├─ input:  title/filename + window_text (title + "\n\n" + window;
        │          first window + sliding windows via the pipeline's
        │          paragraph-aware windower / token windower, 512 overlap)
        ├─ output: doc_type, subclass, calibrated confidence, per-window
        │          votes, agreement, margin, runner-up, OOD flag
        ├─ composite route score S = p × a × m
        └─ rides intake_prep.triage + provenance (source=modernbert)
  4. gate = should_llm_intake(text, stats, ml_triage)
        LLM intake + sorter fire ONLY when:
          • clerk says messy, OR
          • ml_triage missing/failed (fail-open), OR
          • route != fast_path (thresholds/agreement/margin/support unmet),
          • or the pipeline's existing budget/retry rules fire
  5. manifest + catalog + audit               (unchanged)
```

Fast path requires **all** of: clerk clean · supported doc_type (≠
`unknown`) · supported subclass (or legitimately null) · calibrated doc_type
confidence ≥ threshold · calibrated subclass confidence ≥ threshold when
subclass required · window agreement ≥ 0.80 · margin ≥ 0.10 · authentic
training support ≥ 5 rows · no OOD flag. Initial thresholds from config
(`ROUTE_*`), replaced after selective-risk analysis (§8).

---

## 5. Data layer — build on `mailroom-finetune`

- **Load + verify**: pull `mailroom-finetune @ 19720ceb…` (`ground_truth` +
  `default` joined on `filename` — never positional). Verify the 3,302-row /
  55-strata invariants and content_sha256 discipline (guards against the
  mirror ever drifting from the canonical pin).
- **Labels**: `expected` → doc_type; `expected_subclass` → canonical key via
  the vendored dojo normalization (DMR-066) — `Service`/`service`,
  `Co_Branding`, `Joint Venture _ Filing` → `joint_venture`, etc.
- **Title**: subject → exhibit_description → filename (title-wins).
- **Windows**: token-level, 8,192 budget, 512 overlap, BPE round-trip clamp
  (every published window re-tokenizes ≤ 8,192 WITH specials; title survives
  verbatim) — committed and tested logic, ported byte-for-byte.
- **Splits**: baseline = committed 90/10 stratified by doc_type (seed 42,
  RandomState, no sklearn); upgrade = family-grouped (matter_id, group_id,
  relationships/related_document_ids, thread headers where available).
- **Leakage audit**: no filename across splits; test (323) exact-holdout;
  when enrichment rows land, dedup against all 3,302 canonical rows by
  `content_sha256` before anything enters train.
- **Provenance**: per-row `source_corpus` + `source_revision` +
  `purpose` (canonical | train_only) + `label_source` + `label_confidence`;
  manifest.txt byte-deterministic (no timestamps; the publish commit carries
  time — committed discipline).

---

## 6. Augmentation & enrichment strategy (the new pillar)

### 6.1 Taxonomy and ordering

| Tier | Mechanism | When | Cost | Quality |
|---|---|---|---|---|
| 0 | Authentic corpus only (baseline) | always the first, saved, reported | — | gold standard |
| 1 | **Source-matched enrichment | labels exact, distribution identical | insurance + correspondence immediately | ~$0 compute | highest after Tier 0 |
| 2 | Enron pseudo-label distillation | correspondence, after Tier 1 measured | encoder inference only (≈20 min CPU for 247K @ 5 ms/doc) | provenance-marked, confidence-gated |
| 3 | Constrained LLM label-card synthesis | only labels with no authentic floor (CUAD contract tail, corporate_record, merger) | LLM budget per token | full 7-gate pipeline |

**Ladder rule:** every tier is A/B'd against the previous tier on the held-out
test; adopt only where macro-F1 improves AND ECE stays within budget. Nothing
new enters val/test/calibration, ever.

### 6.2 Enron (247,523 dedup)

- **GT-subset expansion (Tier 1):** the `ground_truth` config of
  `enron-correspondence-dedup` carries the enriched labels for the mailroom's
  correspondence surface — rows that share the aeslc_join/llm_zero_shot
  lineage and are NOT already in the corpus become legitimate Tier-1
  training rows (source_revision pinned, purpose=train_only). Cap: ≤ +2× the
  current correspondence train rows.
- **Pseudo-label distillation (Tier 2):** doc_type = correspondence is
  effectively free (emails by construction); the hard label is subclass
  (email/memo/letter/notice/demand/…). A confident snapshot of the Tier 1
  classifier pseudo-labels blind-pool rows; keep only where doc_type
  confidence ≥ 0.95 AND subclass confidence ≥ 0.9 AND agreement ≥ 0.9 →
  `label_source=pseudo_enron`, `label_confidence` recorded, example weight
  0.5, cap ≤ 30% of correspondence train rows, balanced
  stratification by subclass, **thread-grouped split enforced** (thread
  families never straddle train vs val).
- **OOD + drift infrastructure (free):** the blind 247K pool is the
  novelty/OOD probe distribution (OOD flag in the route gate), the
  embedding-diversity baseline for chromatic diversity checks, and the drift
  monitoring pool (predicted-label distribution drift, filename/title drift,
  per-class abstention) — no labels needed.
- **Domain-adaptive MLM (optional P-extra, cost-gated): continued
  masked-LM pretraining on the dedup pool would shift the encoder toward
  mailroom email bodies — only considered if fine-tune plateaus at a hard
  plateau; at 247K docs × 8,192 tokens this is real GPU compute, and
  fine-tuned-small-models doctrine (2406.08660) says the supervised signal, not
  more unlabeled text, is what drives classification. **Default: skip; revisit
  only on plateau.**
- **Messiness (Phase 2): thread bodies with headers/footers, encoding
  artifacts, and quoted replies are natural messy/clean pair signal — a weak
  supervision source for the Phase-2 messy head.

### 6.3 Insurance pools

- **Label mapping (Tier 1): which pool feeds which subclass:

| Subclass | Pool | Verified headroom | Step |
|---|---|---|
| carrier / inpatient / outpatient / pde → 20% add | CMS DE-SynPUF rendered pool + renderer | +400 rendered now; render more only if needed | hashes dedup vs canonical |
| property → 3× | GNOTHEIA polycontexts (863) | +~660 docs, Apache-2.0 | direct (docs are rendered claims) |
| auto → 4× | BDR tabular (800) | decision letters must be re-rendered (corpus v8 had the render step) — if the renderer is reproducible, take all 800; else skip | render + provenance |
| narratives → 10×+ | INSURBIAS CSV (102.6 MB) | count at load; pool thousands | parse + map |
- **Imbalance guard:** insurance is already the majority class (33%). Cap
  total insurance train rows at a plan-set budget (e.g. ≤ 2× class size) and
  let inverse-frequency weights + subclass-aware sampling do the rest;
  enrichment targets **subclass tails** (property, auto, narrative
  subclasses), not raw class volume.
- **License math:** GNOTHEIA (Apache-2.0) + BDR (MIT) + INSURBIAS
  (CC-BY-4.0) + CMS (public-use synthetic) — all permissive; the derived
  prepared set stays CC-BY-4.0 with per-row provenance (same as the
  committed v1 set).
- **Dedup:** content_sha256 + title/near-dup against the 3,302 canonical rows
  and within pool; first-occurrence wins (committed discipline).

### 6.4 Related pools (same mechanism)

`mailroom-cuad-contracts-full`, `mailroom-maud-contracts`,
`mailroom-s1-corporate-records` extend the same Tier-1 story to contract /
merger / corporate-record classes. Structure + license for each verified at
enrichment build time (not asserted here — factuality rule).

### 6.5 Constrained LLM synthesis (Tier 3 — plan B policy, committed verbatim)

- Eligibility: < 20 authentic rows, macro-F1 below floor, high confusion
  with neighbor, or sibling underrepresentation. Never for labels with
  adequate authentic diversity; no equalization for its own sake.
- Caps: 0–4 authentic → no synthetic-only labels (LLM-routed / `other`);
  5–14 → ≤3×; 15–29 → ≤2×; 30–74 → ≤1×; 75+ → none.
- Generation: structured **label cards** (parent class, exact subclass,
  positive/negative cues, title patterns, required structure, target length,
  variation dims, prohibited facts); fictional entities; never paraphrase of
  a training document.
- Seven gates: schema → lexical contamination (n-gram/entity overlap) →
  rule-based cue checks → independent adjudication (a second model, never the
  generator) → embedding-diversity → model-disagreement filter → human spot
  audit of rare classes. Rejects retained in the audit store, never fitted.
- Mixture: ≥60–70% authentic per epoch; ≤40% synthetic globally; ≤50% per
  subclass batch; synthetic example weight 0.6 (0.4–0.7 band); oversample
  authentic minority first.

---

## 7. Training recipe (committed + hardening) — `answerdotai/ModernBERT-base`

| Setting | Value | Source |
|---|---|---|
| Optimizer | AdamW, bf16 on CUDA / fp32 CPU fallback | committed |
| LR | 2e-5 linear, warmup 6%, weight decay 0.01 | committed + plan B |
| Epochs | 3–8, early stop patience 2 | committed |
| Early-stop criterion | val loss (baseline); **val macro-F1 under acceptable ECE** (hardened) | plan B |
| Batch | 16, grad-accum 2 (effective 32) | committed |
| Gradient clipping | 1.0 | plan B |
| Loss | class-weighted CE per head (weights ∝ 1/freq over train); synthetic rows weighted 0.6 | committed + plan B |
| Seeds | 42 + 7 + 2026 for the final candidate (baseline-first discipline: real-only model saved + reported before augmentation) | plan B |
| Checkpoint | best val macro-F1 s.t. ECE acceptable | plan B |
| Test gate | `--eval-test`: held-out doc_type/subclass accuracy (P0: doc_type ≥ 0.95, subclass ≥ 0.75) | committed |
| Compute | Modal L4 (`deploy/modal_app.py`, committed + SDK-refreshed) ≈ 1–2 L4-hours one-time | committed |

Windows per document: title+head window(s) + sampled body windows, document
-level weighting (long docs don't overwhelm); final window optional (signature
blocks carry signal); windows built AFTER the split — windows of one document
never straddle train/validation.

---

## 8. Calibration & selective risk

- Per-head temperature scaling (Platt, bounded scalar, `minimize_scalar`) on
  validation logits; heads with < 2 classes in val left uncalibrated (T=1).
- **Selective-risk analysis (new): a threshold sweep on the calibration/threshold
  set builds the reliability tables + ECE + the conditional error rate among
  fast-pathed documents; the deployment thresholds are the ones that meet
  the error budget (initial config: P(err|fast path) ≤ 0.02; ECE ≤ 0.05 across
  the 0.88–0.97 band). Report shape identical to `calibration:classify`
  (reliability, ECE, review-gate threshold).
- Composite score `S = p × a × m` ride the same span.

---

## 9. Routing integration + observability

- `should_llm_intake(text, stats, ml_triage)` — extends, never replaces: no
  text → False; messy → True; missing/failed ml_triage → True; route !=
  fast_path → True; else False (fast path). LLM intake merges with preserved
  provenance (deterministic | modernbert | llm | merged) so the sorter
  always sees the triage source.
- Fail-open: every ML dependency wrapped; any exception → LLM path, with a
  span recording the failure. Never silently assign.
- Span contract `intake-ml-triage`: input (filename, raw/clean chars,
  windows), output (doc_type, subclass, confidence, route), metadata
  (model_id, artifact_sha, dataset_revision, calibration_version,
  label_schema_version, synthetic_policy_version, window_agreement).
- Metrics per path: fast-path rate, LLM-fallback rate, fast-path accuracy
  from audited outcomes, selective risk, ECE, drift, OOD-rate trend,
  per-class abstention, synthetic-vs-real-label delta.

---

## 10. Serving & cost model

| Path | Latency/doc | Cost/doc | Verdict |
|---|---|---|---|
| **ONNX int8 CPU** (optimum export; ~60 MB; loads in ms) | 5–20 ms | ~$0.000001 | **primary** |
| Modal L4 shared (fallback) | 1–5 ms | amortized | fallback if CPU latency ever matters |
| LLM sorter tail (~10–20% of docs) | 1–3 s | ~$0.0002–0.0004 | deepseek-v4-flash/qwen3.7-flash path |
| Gemma-4-E4B-it (Modal) reference | 2–10 s | ~$0.0005–0.001 | existing comparison runs |

At ~1,000 docs/day steady state: ≥ 80% skip the LLM → LLM spend cut 80–90%.

---

## 11. Evaluation & success criteria (mailroom-evals surfaces)

1. `eval:classification` on the DMR-067 stratified 50 (`--subset test
   --sample 50 --seed 42`): doc_type + subclass accuracy + per-stratum
   confusion — classifier vs deepseek-v4-flash vs gemma-4-E4B-it vs
   qwen3.7-flash (same harness, same seeds — the caller's original ask).
2. `calibration:classify` — reliability + ECE, band-mapping proof.
3. `compare_runs.py --a <classifier> --b <deepseek>` — A/B + CIs.
4. Cost ledger: `performance.by_agent` (classifier logs CPU cost constant).

| Metric | Threshold to deploy |
|---|---|
| doc_type accuracy (stratified 50) | ≥ 0.95 (LLM reference: 0.90–0.96 prior runs) |
| subclass accuracy | ≥ 0.75 (LLM reference: 0.40–0.76 prior runs) |
| ECE | ≤ 0.05 across the 0.88–0.97 band |
| Selective risk | P(err | fast path) ≤ 0.02 (initial) |
| LLM-call reduction | ≥ 80% of documents skip the LLM |
| Cost/doc | ≤ 1/100th of the deepseek-v4-flash path |
| Fail-open | 0 silent mis-routes in shadow week |

---

## 12. Rollout ladder + retrain cadence

1. **Shadow:** classifier on every document, LLM never skipped; log
   agreement/route recommendations (span only). 2. **Restricted fast path:**
   skip only clean + high-confidence + ≥5 authentic support. 3. **Expand by
   evidence:** new labels enter the fast path only under measured selective
   risk. 4. **Retrain cadence:** retrain only with versioned data changes;
   recalibrate every release; previous artifact retained for rollback.
   5. **Human review loop:** low-confidence / OOD / high-impact disagreements
   → review queue; reviewed authentic docs beat unlimited synthesis.

---

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Title-wins doctrine overfits filenames | grouped family split; title+head+body training; held-out eval; per-stratum confusion monitoring |
| Enron thread leakage | thread/header grouping mandatory the moment Enron rows train; near-dup dedup |
| Pseudo-label noise | confidence gates + weight 0.5 + caps + provenance; measured on held-out test |
| Insurance enrichment deepens imbalance | class caps + inverse-frequency weights + subclass-aware sampling; enrichment targets tails |
| Calibration drift | per-release recalibration; `calibration:classify` gate |
| Encoder can't learn corpus conventions (SEC hybrids, etc.) | rules stay in the LLM prompt; cascade defers via low confidence |
| Synthetic shortcuts | 7 gates + mixture caps + adversarial adjudicator |
| ONNX export drift | logits-parity test (PyTorch vs ONNX, ≥ 1e-4 tolerance) before release |
| Garbled/OCR-messy docs | clerk messy flag routes before trust; Phase-2 messy head |
| Training cost | 1–2 L4-hours one-time, amortized; CPU smoke path in repo |
| Mailroom-finetune mirrors drift from canonical | load-time hash + row-count verification (§5) |

---

## 14. Work breakdown (units + evidence gates)

| Unit | Owner | Deliverable | Gate |
|---|---|---|---|
| U1 | orchestrator | this plan + repo skeleton + config interlock | plan reviewed, facts cited |
| U2 | athena-database-agent | data layer: labels/preprocessing/windows/dataset (mailroom-finetune)/grouped split/leakage audit/build CLI + provenance | `uv run pytest tests/` green; deterministic manifest; full-corpus contract test gated |
| U3 | lucius | model/inference (composite score + sorter merge)/calibration/selective risk/routing/fail-open/tracing contract + training CLI + eval CLI | toy-CPU smoke; routing unit tests incl. fail-open; calibration math tests |
| U4 | lucius | augmentation: enrichment assembler (Tier 1 Enron GT + insurance pools), pseudo-label pipeline scaffold, label-card generator + 7 gates (Tier 3) | gates unit-tested; every adopted tier must have a ladder step + provenance |
| U5 | modal-specialist | deploy/modal_app.py (SDK-current) + serve app contract | modal deploy dry-run/validation |
| U6 | orchestrator | integration, CPU smoke run, report | all gates green, report written |

---

## 15. Sources

1. `eval-environment/docs/intake-classifier-proposal.md` (plan A)
2. `ModernBERT implementation.md` (plan B, ppl-ai file)
3. Mailroom-Corpus-EDA @ `cf096fa` — committed modernbert tooling + tests
4. HuggingFace Hub API (live, 2026-09-18): model card `answerdotai/ModernBERT-base`;
   repos `mailroom-dataset`, `mailroom-finetune`, `mailroom-modernbert-training`,
   `enron-correspondence-dedup`, `cms-desynpuf-insurance-claims`,
   `gratex/GNOTHEIA-synthetic-insurance-dataset`,
   `bdr-ai-org/insurance-motor-claims-decision-v1`, `feihuangfh/INSURBIAS`;
   papers 2406.08660, 2412.13663 (metadata verified)