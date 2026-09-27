# TEST-EVAL — Arm B primary (`m9a-local-20260927-014429`)

**Eval role:** authoritative held-out test for M9a **#112** gates and Hub release  
**Run tag:** `m9a-local-20260927-014429`  
**Run ID:** `20260927-064513`  
**Arm:** B (primary, GPU 0)  
**Checkpoint:** `data/modernbert_training/runs/m9a-local-20260927-014429/latest`  
**Eval JSON (this dir):** `reports/TEST-EVAL/eval_m9a-local-20260927-014429.json`  
**Eval JSON (canonical):** `reports/eval_m9a-local-20260927-014429.json`  
**Artifact SHA:** `faf97878b05a3426a787e8cea55f8b835e1010978cc7844e2a7c43dce09d78bc`  
**Training log:** `logs/m9a-local-20260927-014429.log`  
**Training report:** `reports/M9a-REPORT-m9a-local-20260927-014429.md`

## Training config (context for test readout)

| Setting | Value |
| --- | --- |
| `loss_lambda_dt` | **0.65** (Arm B) |
| epochs | 3 |
| batch × grad_accum | 4 × 8 |
| lr | 2e-5 |
| seed | 42 |
| max_length | 8192 |
| label_smoothing | 0.05 (doc_type CE only) |
| weight_mode / cap | inverse / 20 |
| warmup_frac | 0.06 |
| mlp_heads | true |
| freeze_backbone_epochs | 0 |
| model | `answerdotai/ModernBERT-base` |
| data | `data/modernbert_training/stage` |
| training wall time | ~4.5 h (16,326 s) |

## Test harness protocol

| Parameter | Value |
| --- | --- |
| Command surface | `training/eval_modernbert.py` |
| `--subset` | `test` |
| Documents | **323** (full held-out split, `--sample 0`) |
| `--max-length` | 8192 |
| `--seed` | 42 |
| Windows evaluated | 541 (`window_calibration.n_windows`) |

The test split is isolated from training and from validation checkpoint
selection (plan D11). Report-only **P0** thresholds in `recorded_gates`
(doc_type ≥ 0.95, subclass ≥ 0.75) are diagnostic; **#112** gates below are
the release bar.

## Headline test metrics (eval harness)

| Metric | Value |
| --- | ---: |
| doc_type_accuracy | **0.9505** (307/323) |
| subclass_accuracy_conditional | **0.5831** (179/307 scorable) |
| window_calibration ECE | **0.0259** |
| window_calibration band ECE | 0.0525 |
| fast_path_rate | 0.3034 |

### Trainer vs eval harness

Validation **epoch 3** was shipped with trainer **`gate_met: true`** (calibrated
doc_type ECE ≤ 0.05, no macro-F1 regression beyond 0.005). For **held-out test**,
use this eval JSON — not inline `summary.json` test fields alone — when judging
#112 or updating the Hub model card.

| Surface | doc_type | subclass (cond.) | window ECE |
| --- | ---: | ---: | ---: |
| Val epoch 3 (trainer) | acc 0.9331; macro-F1 0.9389 | objective 0.3334 | ECE 0.02 (cal.) |
| **Held-out test (harness)** | **0.9505** | **0.5831** | **0.0259** |

Test doc_type accuracy **exceeds** validation doc accuracy (0.9331), consistent
with a strong routing head and no test leakage (fixed split).

## Cohorts (#104 single vs multi-window)

| cohort | n_docs | doc_type_accuracy | mean_agreement | window ECE |
| --- | ---: | ---: | ---: | ---: |
| single-window | 274 | 0.9489 | 1.0 | 0.0426 |
| multi-window | 49 | 0.9592 | 0.9724 | 0.0168 |

Multi-window docs show **higher** doc_type accuracy and **lower** window ECE than
single-window on this checkpoint; mean agreement remains high (0.9724) on the
49 multi-window cases.

## Selective-risk sweep (global)

| Field | Value |
| --- | --- |
| budget_met | **true** |
| recommended deployment threshold | **0.90** |
| error_budget | 0.02 |
| coverage at pick | ~0.377 |

Per-cohort selective-risk on multi-window reports `budget_met: false` with
`recommended_threshold: null` (insufficient sweep pick under stricter cohort
constraints); global sweep is the deployment reference documented in the primary
M9a training report.

## Per-head test macro-F1 (observed)

Support counts are per-class **test** support from `per_head.<head>.support`
(#112 M9a-U4 — read macro-F1 beside support for sparse heads).

| head | macro-F1 | notes |
| --- | ---: | --- |
| insurance_claim | 0.9889 | near-balanced control head |
| corporate_record | 0.2485 | partial learning |
| merger_agreement | 0.2381 | partial learning |
| correspondence | 0.1289 | below #112 floor |
| contract | 0.0195 | majority-prior collapse persists |

### Calibrated head ECE (validation / exclusion policy)

From trainer checkpoint selection (validation, not test):

| head | ECE (calibrated) | fast-path excluded |
| --- | ---: | --- |
| doc_type | 0.02 | no |
| insurance_claim | 0.0025 | no |
| contract | 0.0327 | no |
| corporate_record | 0.071 | **yes** |
| correspondence | 0.1384 | **yes** |
| merger_agreement | 0.1083 | **yes** |

High-ECE subclass heads stay off the fast path even when doc_type routing is strong.

## M9a #112 gates (held-out test)

Verified with `training/check_m9a_gates.py` → `gates-check-armB.txt`.

| Gate | Actual | Threshold | Status |
| --- | ---: | ---: | --- |
| contract test macro-F1 | 0.0195 | ≥ 0.20 | **NOT MET** |
| correspondence test macro-F1 | 0.1289 | ≥ 0.25 | **NOT MET** |
| doc_type test accuracy | 0.9505 | ≥ 0.89 | **MET** |
| window ECE (doc_type calibrated) | 0.0259 | ≤ 0.05 | **MET** |

**Overall #112 gate verdict:** **FAIL** (2/4 gates not met — contract and
correspondence subclass heads remain below Verify-grade thresholds; doc_type and
calibration gates pass).

### Report-only P0 (eval harness)

| Gate | Actual | Threshold | met |
| --- | ---: | ---: | --- |
| P0 doc_type | 0.9505 | 0.95 | true |
| P0 subclass | 0.5831 | 0.75 | false |

## Baseline comparison (doc_type test accuracy)

| source | doc_type_accuracy |
| ---: | ---: |
| run-3 (`eval_run3_20260921.json`) | 0.8947 |
| **this run (Arm B)** | **0.9505** |
| Arm A smoke | 0.8638 |

Δ vs run-3: **+5.58 pp** doc_type; conditional subclass 0.5831 vs run-3 0.5260
(+5.7 pp).

## Hub release

| Field | Value |
| --- | --- |
| Model repo | `Lucius-Morningstar/mailroom-modernbert-classifier` |
| Release tag | `m9a-local-20260927-014429` |
| Eval on Hub | `eval_report_m9a-local-20260927-014429.json` |

Publish via `./training/complete_run.sh --run-tag m9a-local-20260927-014429 --publish`
(requires all #112 gates MET, or `--force-publish` to record partial gate state).

## TEST-EVAL verdict

- **Routing (doc_type):** production candidate — **0.9505** test accuracy, ECE
  **0.0259**, passes #112 doc_type and calibration gates; large gain vs run-3.
- **Subclass (#112):** **FAIL** — contract and correspondence macro-F1 unchanged
  in spirit from the M9a diagnosis (collapse / prior), despite improved
  conditional subclass accuracy vs run-3 and vs Arm A smoke.
- **Recommended default:** this checkpoint over the 1-epoch Arm A ablation.
