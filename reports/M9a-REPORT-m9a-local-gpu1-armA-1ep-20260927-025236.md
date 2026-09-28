# M9a local A — full report

**Run tag:** `m9a-local-gpu1-armA-1ep-20260927-025236`  
**Arm:** A  
**Checkpoint:** `data/modernbert_training/runs/m9a-local-gpu1-armA-1ep-20260927-025236/latest`  
**Held-out eval JSON:** `reports/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json`

## Held-out test (323 docs @ 8,192 tokens)

Source: `reports/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json`.

| Metric | Value |
| --- | ---: |
| doc_type_accuracy | **0.8638** (279/323) |
| subclass_accuracy_conditional | **0.4839** (135/279 scorable) |
| window_calibration ECE | **0.0397** |
| window_calibration band ECE | 0.0574 |

## Cohorts (#104 single vs multi-window)

| cohort | n_docs | doc_type_accuracy | mean_agreement | window ECE |
| --- | ---: | ---: | ---: | ---: |
| single-window | 274 | 0.8723 | 1.0000 | 0.1179 |
| multi-window | 49 | 0.8163 | 0.8020 | 0.1024 |

## Selective-risk sweep (global)

| Field | Value |
| --- | --- |
| budget_met | **None** |
| recommended deployment threshold | **None** |
| error_budget | None |
| coverage at pick | ~— |

Per-cohort selective-risk on multi-window reports `budget_met: None` with `recommended_threshold: None`.

### Per-head test macro-F1 (observed)

| head | macro-F1 |
| --- | ---: |
| contract | 0.0039 |
| corporate_record | 0.0212 |
| correspondence | 0.0944 |
| insurance_claim | 0.7552 |
| merger_agreement | 0.1600 |

## M9a #112 gates (held-out test)

Verified with `training/check_m9a_gates.py` → `reports/TEST-EVAL/gates-check-armA.txt`.

| Gate | Actual | Threshold | Status |
| --- | ---: | ---: | --- |
| contract test macro-F1 | 0.0039 | ≥ 0.2 | **NOT MET** |
| correspondence test macro-F1 | 0.0944 | ≥ 0.25 | **NOT MET** |
| doc_type test accuracy | 0.8638 | ≥ 0.89 | **NOT MET** |
| window ECE (doc_type calibrated) | 0.0397 | ≤ 0.05 | MET |

**Overall #112 gate verdict:** **FAIL** (3/4 gates not met).

### Report-only P0 (eval harness)

| Gate | Actual | Threshold | met |
| --- | ---: | ---: | --- |
| P0 doc type | 0.8638 | 0.95 | False |
| P0 subclass | 0.4839 | 0.75 | False |

## Hub release

- **Model repo:** `Lucius-Morningstar/mailroom-modernbert-classifier`  
- **Release tag:** `m9a-local-gpu1-armA-1ep-20260927-025236`  
- **Eval artifact on Hub:** `eval_report_m9a-local-gpu1-armA-1ep-20260927-025236.json`  

## Verdict

- **Doc-type routing:** test acc 0.8638, ECE 0.0397.
- **#112 gates:** FAIL.
