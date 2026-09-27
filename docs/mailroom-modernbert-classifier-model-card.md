---
library_name: transformers
pipeline_tag: text-classification
base_model: answerdotai/ModernBERT-base
datasets:
- Lucius-Morningstar/mailroom-modernbert-training
language:
- en
tags:
- modernbert
- document-classification
- hierarchical
- legal
- mailroom
- m9a
license: apache-2.0
model-index:
- name: m9a-local-20260927-014429
  results:
  - task:
      type: text-classification
      name: doc_type held-out test (323 docs)
    dataset:
      name: mailroom held-out test
      type: Lucius-Morningstar/mailroom-modernbert-training
    metrics:
    - type: accuracy
      value: 0.9505
      name: doc_type_accuracy
    - type: accuracy
      value: 0.5831
      name: subclass_accuracy_conditional
---

# mailroom-modernbert-classifier

Hierarchical document classifier for the **LLM-Mailroom** intake pipeline: a
fine-tuned **ModernBERT-base** encoder with a `doc_type` head plus one
subclass head per document class. It is the deterministic pre-check in the
BERT-coupled intake overhaul (mailroom-issues #85).

**Recommended production revision (M9a primary, Arm B):**
[`m9a-local-20260927-014429`](https://huggingface.co/Lucius-Morningstar/mailroom-modernbert-classifier/tree/m9a-local-20260927-014429)
— 3 epochs, `loss_lambda_dt=0.65`, held-out `doc_type` accuracy **0.9505**,
trainer validation **`gate_met: true`**. M9a mailroom-issues **#112** subclass
gates remain **not met** for `contract` / `correspondence`; use LLM sorter for
subclass-ambiguous tails.

## Hub releases (M9a, 2026-09-27)

| Release tag | Arm | Epochs | λ_dt | Held-out doc_type acc | Held-out subclass (cond.) | Trainer `gate_met` | #112 gates | Eval on Hub |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| [`m9a-local-20260927-014429`](https://huggingface.co/Lucius-Morningstar/mailroom-modernbert-classifier/tree/m9a-local-20260927-014429) | **B (primary)** | 3 | 0.65 | **0.9505** | **0.5831** | **true** | FAIL (2/4) | [`eval_report_m9a-local-20260927-014429.json`](https://huggingface.co/Lucius-Morningstar/mailroom-modernbert-classifier/blob/main/eval_report_m9a-local-20260927-014429.json) |
| [`m9a-local-gpu1-armA-1ep-20260927-025236`](https://huggingface.co/Lucius-Morningstar/mailroom-modernbert-classifier/tree/m9a-local-gpu1-armA-1ep-20260927-025236) | A (smoke) | 1 | 0.50 | 0.8638 | 0.4839 | false | FAIL (3/4) | [`eval_report_m9a-local-gpu1-armA-1ep-20260927-025236.json`](https://huggingface.co/Lucius-Morningstar/mailroom-modernbert-classifier/blob/main/eval_report_m9a-local-gpu1-armA-1ep-20260927-025236.json) |

Held-out test set: **323 documents**, 8,192-token windows, never used for
training, calibration fit, or threshold tuning. Authoritative metrics come from
the `eval_report_<tag>.json` artifacts (not inline `summary.json` test fields
when they differ).

Contractor-repo report mirrors (same content as sections below):
[`M9a-REPORT-m9a-local-20260927-014429`](https://github.com/LLM-Mailroom-Services/mailroom-ml/blob/main/reports/M9a-REPORT-m9a-local-20260927-014429.md),
[`M9a-REPORT-m9a-local-gpu1-armA-1ep-20260927-025236`](https://github.com/LLM-Mailroom-Services/mailroom-ml/blob/main/reports/M9a-REPORT-m9a-local-gpu1-armA-1ep-20260927-025236.md),
[`M9a-COMPARE-ArmA-vs-ArmB-20260927`](https://github.com/LLM-Mailroom-Services/mailroom-ml/blob/main/reports/M9a-COMPARE-ArmA-vs-ArmB-20260927.md).

## Architecture

- **Backbone:** `answerdotai/ModernBERT-base` — 22 layers, 768 hidden,
  8,192-token context, bf16.
- **Heads:** one head per class — `doc_type` (6 classes) plus 5 subclass
  heads (`contract`, `corporate_record`, `correspondence`, `insurance_claim`,
  `merger_agreement`). MLP heads with dropout 0.1.
- **Windowing:** token-level, 8,192 tokens with 512-token overlap. `doc_type`
  by plurality vote over windows; subclass by plurality over windows whose
  `doc_type` vote is the winning class.
- **Calibration:** per-head temperature scaling (`temperatures.json`).

## Files

| file | purpose |
|---|---|
| `model.safetensors` | ModernBERT backbone weights (bf16, ~298 MB) |
| `heads.pt` | hierarchical head state dicts |
| `labels.json` | head vocabularies (`labels` / `label2id` / `id2label` / `weights`) |
| `temperatures.json` | per-head calibration temperatures |
| `train_counts.json` | per-(doc_type, subclass) authentic train-row counts (support gate) |
| `config.json`, `tokenizer.json`, `tokenizer_config.json` | backbone config + tokenizer |
| `summary.json` | full run summary (hyperparameters, per-epoch metrics, selection, test metrics) |
| `eval_report_<release_tag>.json` | held-out test harness output for that release (on repo root) |

## Labels

**doc_type (6):** `contract`, `merger_agreement`, `corporate_record`,
`correspondence`, `insurance_claim`, `unknown` *(inference-only abstention —
not a trained class)*

**contract (24):** agency, co_branding, collaboration, consulting,
development, distributor, endorsement, franchise, hosting, ip, joint_venture,
license, maintenance, manufacturing, marketing, other, outsourcing, promotion,
reseller, service, sponsorship, strategic_alliance, supply, transportation

**corporate_record (10):** articles_of_incorporation, board_resolution,
bylaws, charter_amendment, indenture, officer_certificate, other,
powers_of_attorney, rights_instrument, subsidiary_list

**correspondence (7):** demand, email, letter, meeting_request, memo, notice,
press_release

**insurance_claim (6):** auto, carrier, inpatient, outpatient, pde, property

**merger_agreement (5):** all_cash, all_stock, mixed_cash_stock,
mixed_cash_stock_election, other

## Usage

The backbone is a standard `ModernBertModel`; the hierarchical heads are a
custom bundle. Load a **pinned release tag** (never a floating `main` tip in
production):

```python
from huggingface_hub import snapshot_download
from mailroom_ml.inference import load_bundle, classify_document

model_dir = snapshot_download(
    "Lucius-Morningstar/mailroom-modernbert-classifier",
    revision="m9a-local-20260927-014429",
)
bundle = load_bundle(model_dir)
result = classify_document(
    title="Notice of Default",
    text=document_text,
    bundle=bundle,
)
# result["doc_type"], result["subclass"], result["confidence"], result["route"]
```

`route == "fast_path"` means the calibrated gate passed and the LLM sorter may
be skipped (skip mode + allowlist only); otherwise route the document to the
LLM sorter with the BERT triage as an advisory prior.

## Limitations

- **Subclass heads are weak** for `contract` / `correspondence` /
  `corporate_record` / `merger_agreement` on held-out test (M9a Arm B macro-F1
  0.02–0.25). Use the `doc_type` head for routing; route subclass-ambiguous
  documents to the LLM sorter. Do **not** trust skip-mode subclass for these
  classes until #112 gates pass.
- Trained on a curated legal-document corpus; not a substitute for legal
  review.
- `unknown` is an inference-only abstention label, not a trained class.
- Long documents are windowed (8,192 tokens, 512 overlap); oversize documents
  fall back to the LLM path when routing policy requires it.

## License

Apache-2.0 (inherits `answerdotai/ModernBERT-base`).

---

# M9a full run report (2026-09-27)

The sections below merge the contractor-repo M9a reports published after local
GPU training. **Arm B** is the production candidate; **Arm A** is a one-epoch
λ ablation smoke on GPU 1.

## M9a local Arm B — full report

**Run tag:** `m9a-local-20260927-014429`  
**Run ID:** `20260927-064513`  
**Arm:** B (primary, GPU 0)  
**Checkpoint:** `data/modernbert_training/runs/m9a-local-20260927-014429/latest`  
**Training log:** `logs/m9a-local-20260927-014429.log`  
**Held-out eval JSON:** `reports/eval_m9a-local-20260927-014429.json` → Hub
`eval_report_m9a-local-20260927-014429.json`

### Config

| Setting | Value |
| --- | --- |
| `loss_lambda_dt` | **0.65** (Arm B) |
| epochs | 3 |
| batch × grad_accum | 4 × 8 |
| lr | 2e-5 |
| seed | 42 |
| max_length | 8192 |
| label_smoothing | 0.05 |
| weight_mode / cap | inverse / 20 |
| warmup_frac | 0.06 |
| mlp_heads | true |
| freeze_backbone_epochs | 0 |
| model | `answerdotai/ModernBERT-base` |
| data | `Lucius-Morningstar/mailroom-modernbert-training` @ prepared stage pin |
| training wall time | ~4.5 h (16,326 s) |

### Validation (per epoch, observed doc_type macro-F1)

| Epoch | doc_type macro-F1 | doc_type doc acc | subclass objective | selected |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.9254 | 0.9164 | — | yes (interim) |
| 2 | 0.9345 | 0.9298 | — | yes (interim) |
| 3 | **0.9389** | **0.9331** | **0.3334** | **yes (shipped)** |

#### Trainer checkpoint selection (validation)

- **Selected epoch:** 3  
- **Selection `gate_met`:** **true** (calibrated doc_type ECE ≤ 0.05, no doc_type macro-F1 regression beyond 0.005)  
- **Val doc_type macro-F1 (observed):** 0.9389  
- **Val doc_type ECE (calibrated):** 0.02  
- **Val subclass objective:** 0.3334  

#### Head exclusion policy (ECE > 0.05 → fast-path excluded)

| head | ECE (calibrated) | excluded |
| --- | ---: | --- |
| doc_type | 0.02 | no |
| contract | 0.0327 | no |
| insurance_claim | 0.0025 | no |
| corporate_record | 0.071 | **yes** |
| correspondence | 0.1384 | **yes** |
| merger_agreement | 0.1083 | **yes** |

### Held-out test (323 docs @ 8,192 tokens)

| Metric | Value |
| --- | ---: |
| doc_type_accuracy | **0.9505** (307/323) |
| subclass_accuracy_conditional | **0.5831** (179/307 scorable) |
| window_calibration ECE | **0.0259** |
| window_calibration band ECE | 0.0525 |

#### Cohorts (#104 single vs multi-window)

| cohort | n_docs | doc_type_accuracy | mean_agreement | window ECE |
| --- | ---: | ---: | ---: | ---: |
| single-window | 274 | 0.9489 | 1.0 | 0.0426 |
| multi-window | 49 | 0.9592 | 0.9724 | 0.0168 |

#### Selective-risk sweep (global)

- **budget_met:** true  
- **recommended deployment threshold:** 0.90  

#### Per-head test macro-F1 (observed)

| head | macro-F1 |
| --- | ---: |
| insurance_claim | 0.9889 |
| corporate_record | 0.2485 |
| merger_agreement | 0.2381 |
| correspondence | 0.1289 |
| contract | 0.0195 |

### M9a #112 gates (held-out test)

| Gate | Actual | Threshold | Status |
| --- | ---: | ---: | --- |
| contract test macro-F1 | 0.0195 | ≥ 0.20 | **NOT MET** |
| correspondence test macro-F1 | 0.1289 | ≥ 0.25 | **NOT MET** |
| doc_type test accuracy | 0.9505 | ≥ 0.89 | **MET** |
| window ECE (doc_type calibrated) | 0.0259 | ≤ 0.05 | **MET** |

**Overall #112 gate verdict:** **FAIL** (2/4 gates not met — contract and
correspondence subclass heads remain below Verify-grade thresholds; doc_type and
calibration gates pass).

### Arm B verdict

- **Doc-type routing:** strong improvement vs run-3 baseline (0.8947 → **0.9505**
  test acc; ECE **0.0259**).  
- **Trainer selection:** epoch 3 **`gate_met` true** on validation.  
- **Subclass / #112:** contract + correspondence test macro-F1 still block full
  #112 PASS; exclusion policy keeps high-ECE subclass heads off the fast path.  
- **Recommended production pointer:** this run (Arm B, λ_dt=0.65) over the
  1-epoch Arm A smoke.

---

## M9a GPU-1 Arm A smoke — full report

**Run tag:** `m9a-local-gpu1-armA-1ep-20260927-025236`  
**Run ID:** `20260927-075323`  
**Arm:** A (one-time GPU 1 smoke per `logs/.m9a-gpu-policy`)  
**Checkpoint:** `data/modernbert_training/runs/m9a-local-gpu1-armA-1ep-20260927-025236/latest`  
**Training log:** `logs/m9a-local-gpu1-armA-1ep-20260927-025236.log`  
**Held-out eval JSON:** `reports/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json` → Hub
`eval_report_m9a-local-gpu1-armA-1ep-20260927-025236.json`  
Published with **`--force-publish`** (smoke compare; gates not required).

### Config

| Setting | Value |
| --- | --- |
| `loss_lambda_dt` | **0.5** (Arm A) |
| epochs | 1 |
| batch × grad_accum | 4 × 8 |
| lr | 2e-5 |
| seed | 42 |
| max_length | 8192 |
| label_smoothing | 0.05 |
| weight_mode / cap | inverse / 20 |
| warmup_frac | 0.06 |
| mlp_heads | true |
| model | `answerdotai/ModernBERT-base` |
| data | `Lucius-Morningstar/mailroom-modernbert-training` @ prepared stage pin |
| training wall time | ~1.55 h (5,579 s) |

### Validation (epoch 1)

| Epoch | doc_type macro-F1 | doc_type doc acc | subclass objective | selection gate_met |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.8158 | 0.8528 | 0.238 | false |

No checkpoint met the lexicographic selection rule (calibrated doc_type ECE
**0.0524** > 0.05 budget on the sole epoch). Trainer reports **`gate_met`
false** and selection epoch 0 (placeholder).

### Held-out test (323 docs @ 8,192 tokens)

| Metric | Value |
| --- | ---: |
| doc_type_accuracy | **0.8638** (279/323) |
| subclass_accuracy_conditional | **0.4839** |
| window_calibration ECE | **0.0397** |
| window_calibration band ECE | 0.0574 |

*Note:* inline trainer `test_metrics.doc_type_acc` (**0.8669**) reflects a
lighter in-train test pass; the authoritative held-out harness is the eval JSON
above.

#### Cohorts

| cohort | n_docs | doc_type_accuracy | mean_agreement | window ECE |
| --- | ---: | ---: | ---: | ---: |
| single-window | 274 | 0.8723 | 1.0 | 0.1179 |
| multi-window | 49 | 0.8163 | 0.802 | 0.1024 |

Selective-risk sweep **refused** (no per-head ECE sidecar in artifact for this
1-epoch smoke).

#### Per-head test macro-F1 (observed)

| head | macro-F1 |
| --- | ---: |
| insurance_claim | 0.7552 |
| merger_agreement | 0.16 |
| correspondence | 0.0944 |
| corporate_record | 0.0212 |
| contract | 0.0039 |

### M9a #112 gates (held-out test)

| Gate | Actual | Threshold | Status |
| --- | ---: | ---: | --- |
| contract test macro-F1 | 0.0039 | ≥ 0.20 | **NOT MET** |
| correspondence test macro-F1 | 0.0944 | ≥ 0.25 | **NOT MET** |
| doc_type test accuracy | 0.8638 | ≥ 0.89 | **NOT MET** |
| window ECE (doc_type calibrated) | 0.0397 | ≤ 0.05 | **MET** |

**Overall #112 gate verdict:** **FAIL** (expected for 1-epoch λ_dt=0.5 compare arm).

### Arm A verdict

Arm A smoke confirms **λ_dt=0.5 + 1 epoch** under-trains doc_type vs Arm B:
−8.7 pp test doc_type acc and −10 pp conditional subclass acc vs the 3-epoch
primary. Use for ablation only; do not promote to production default.

---

## M9a Arm A vs Arm B — comparison

| | **Arm B (primary)** | **Arm A (smoke)** |
| --- | --- | --- |
| Run tag | `m9a-local-20260927-014429` | `m9a-local-gpu1-armA-1ep-20260927-025236` |
| GPU policy | GPU 0 | one-time GPU 1 exception |
| `loss_lambda_dt` | **0.65** | **0.5** |
| epochs | **3** | **1** |
| Eval JSON (Hub) | `eval_report_m9a-local-20260927-014429.json` | `eval_report_m9a-local-gpu1-armA-1ep-20260927-025236.json` |

### Held-out test (n=323)

| Metric | Arm B | Arm A | Δ (B − A) |
| --- | ---: | ---: | ---: |
| doc_type_accuracy | **0.9505** | 0.8638 | **+0.0867** |
| subclass_accuracy_conditional | **0.5831** | 0.4839 | **+0.0992** |
| window ECE | 0.0259 | 0.0397 | −0.0138 (better B) |

### Validation selection

| | Arm B | Arm A |
| --- | --- | --- |
| Selected epoch | 3 | 0 (none qualified) |
| Selection `gate_met` | **true** | false |
| Val doc_type macro-F1 | 0.9389 | 0.8158 (e1 only) |
| Val doc_type ECE (cal.) | 0.02 | 0.0524 (e1) |

### M9a #112 gates (held-out test)

| Gate | Arm B | Arm A |
| --- | --- | --- |
| contract macro-F1 ≥ 0.20 | NOT MET | NOT MET |
| correspondence macro-F1 ≥ 0.25 | NOT MET | NOT MET |
| doc_type acc ≥ 0.89 | **MET** | NOT MET |
| window ECE ≤ 0.05 | **MET** | **MET** |
| **Overall** | **FAIL (2/4)** | **FAIL (3/4)** |

### Interpretation

1. **λ_dt and depth:** Raising doc-type loss weight to 0.65 and training 3
   epochs materially improves both routing accuracy and conditional subclass
   accuracy on the held-out set.  
2. **Production vs smoke:** Arm B is the candidate release; Arm A documents the
   cost of under-training for gate and λ ablation.  
3. **Remaining gap:** Both arms fail contract/correspondence subclass macro-F1
   gates; Arm B additionally passes doc_type accuracy and calibration gates that
   Arm A misses.  
4. **Hub:** Both checkpoints are tagged on this repo with distinct release tags
   and bundled `eval_report_<tag>.json` artifacts.

### Reference baselines (doc_type test accuracy)

| doc_type test acc | source |
| ---: | --- |
| 0.8947 | run-3 (`eval_report_run3.json` / `reports/eval_run3_20260921.json`) |
| **0.9505** | **Arm B (M9a primary)** |
| 0.8638 | Arm A smoke |

---

## Historical — run-3 (2026-09-21)

Checkpoint `runs/20260921-132753` (run_id `20260921-093211`). Config: epochs=2,
batch=4, grad-accum=8, lr=2e-5, `loss_lambda_dt=0.65`, sqrt-inverse weights,
MLP heads. Held-out test doc_type accuracy **0.8947**, conditional subclass
**0.526**, window ECE **0.0203**. Superseded for routing by M9a Arm B
(**0.9505** doc_type acc).

---

<!-- mailroom-ml:test-metrics:begin -->
## Release `m9a-local-20260927-014429` (2026-09-27)

Held-out test eval artifact on this repo: `eval_report_m9a-local-20260927-014429.json`.

### M9a #112 gates (held-out test)

| Gate | Actual | Threshold | Status |
| --- | ---: | ---: | --- |
| contract test macro-F1 | 0.0195 | ≥ 0.2 | NOT MET |
| correspondence test macro-F1 | 0.1289 | ≥ 0.25 | NOT MET |
| doc_type test accuracy | 0.9505 | ≥ 0.89 | MET |
| window ECE (doc_type calibrated) | 0.0259 | ≤ 0.05 | MET |

### Test surfaces

- subclass_accuracy_conditional: 0.5831
- window_calibration: ece=0.0259 band_ece=0.0525

### Per-head test macro-F1 (observed)

| head | macro-F1 |
| --- | ---: |
| contract | 0.0195 |
| corporate_record | 0.2485 |
| correspondence | 0.1289 |
| insurance_claim | 0.9889 |
| merger_agreement | 0.2381 |

### Trainer selection (validation)

- selected epoch: 3
- gate_met: True
- subclass_objective: 0.33340000000000003
- doc_type macro-F1 (val): 0.9389
- doc_type ECE (val, calibrated): 0.02
- head_exclusion_policy (ECE > 0.05):
  - contract: excluded=False ece=0.0327
  - corporate_record: excluded=True ece=0.071
  - correspondence: excluded=True ece=0.1384
  - doc_type: excluded=False ece=0.02
  - insurance_claim: excluded=False ece=0.0025
  - merger_agreement: excluded=True ece=0.1083

<!-- mailroom-ml:test-metrics:end -->
