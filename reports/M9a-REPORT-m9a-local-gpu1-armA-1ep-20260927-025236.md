# M9a GPU-1 Arm A smoke — full report (2026-09-27)

**Run tag:** `m9a-local-gpu1-armA-1ep-20260927-025236`  
**Run ID:** `20260927-075323`  
**Arm:** A (one-time GPU 1 smoke per `logs/.m9a-gpu-policy`)  
**Checkpoint:** `data/modernbert_training/runs/m9a-local-gpu1-armA-1ep-20260927-025236/latest`  
**Training log:** `logs/m9a-local-gpu1-armA-1ep-20260927-025236.log`  
**Held-out eval JSON:** `reports/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json`

## Config

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
| data | `data/modernbert_training/stage` |
| training wall time | ~1.55 h (5,579 s) |

## Validation (epoch 1)

| Epoch | doc_type macro-F1 | doc_type doc acc | subclass objective | selection gate_met |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.8158 | 0.8528 | 0.238 | false |

No checkpoint met the lexicographic selection rule (calibrated doc_type ECE **0.0524** > 0.05 budget on the sole epoch). Trainer reports **`gate_met` false** and selection epoch 0 (placeholder).

## Held-out test (323 docs @ 8,192 tokens)

Source: `reports/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json`.

| Metric | Value |
| --- | ---: |
| doc_type_accuracy | **0.8638** (279/323) |
| subclass_accuracy_conditional | **0.4839** |
| window_calibration ECE | **0.0397** |
| window_calibration band ECE | 0.0574 |

*Note:* inline trainer `test_metrics.doc_type_acc` (0.8669) reflects a lighter in-train test pass; the authoritative held-out harness is the eval JSON above.

### Cohorts

| cohort | n_docs | doc_type_accuracy | mean_agreement | window ECE |
| --- | ---: | ---: | ---: | ---: |
| single-window | 274 | 0.8723 | 1.0 | 0.1179 |
| multi-window | 49 | 0.8163 | 0.802 | 0.1024 |

Selective-risk sweep **refused** (no per-head ECE sidecar in artifact for this 1-epoch smoke).

### Per-head test macro-F1 (observed)

| head | macro-F1 |
| --- | ---: |
| insurance_claim | 0.7552 |
| merger_agreement | 0.16 |
| correspondence | 0.0944 |
| corporate_record | 0.0212 |
| contract | 0.0039 |

## M9a #112 gates (held-out test)

| Gate | Actual | Threshold | Status |
| --- | ---: | ---: | --- |
| contract test macro-F1 | 0.0039 | ≥ 0.20 | **NOT MET** |
| correspondence test macro-F1 | 0.0944 | ≥ 0.25 | **NOT MET** |
| doc_type test accuracy | 0.8638 | ≥ 0.89 | **NOT MET** |
| window ECE (doc_type calibrated) | 0.0397 | ≤ 0.05 | **MET** |

**Overall #112 gate verdict:** **FAIL** (expected for 1-epoch λ_dt=0.5 compare arm).

## Hub release

- **Model repo:** `Lucius-Morningstar/mailroom-modernbert-classifier`  
- **Release tag:** `m9a-local-gpu1-armA-1ep-20260927-025236`  
- **Eval artifact on Hub:** `eval_report_m9a-local-gpu1-armA-1ep-20260927-025236.json`  
- Published with **`--force-publish`** (smoke compare; gates not required).

## Verdict

Arm A smoke confirms **λ_dt=0.5 + 1 epoch** under-trains doc_type vs Arm B: −8.7 pp test doc_type acc and −10 pp conditional subclass acc vs the 3-epoch primary. Use for ablation only; do not promote to production default.
