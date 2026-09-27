# TEST-EVAL — Arm A smoke (`m9a-local-gpu1-armA-1ep-20260927-025236`)

**Eval role:** ablation / compare arm — **not** production default  
**Run tag:** `m9a-local-gpu1-armA-1ep-20260927-025236`  
**Run ID:** `20260927-075323`  
**Arm:** A (one-time GPU 1 smoke per `logs/.m9a-gpu-policy`)  
**Checkpoint:** `data/modernbert_training/runs/m9a-local-gpu1-armA-1ep-20260927-025236/latest`  
**Eval JSON (this dir):** `reports/TEST-EVAL/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json`  
**Eval JSON (canonical):** `reports/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json`  
**Artifact SHA:** `5ecea781599be293dcc80f32ac4b57b478d6787aed2c7bbf2cc2a01858b60aae`  
**Training log:** `logs/m9a-local-gpu1-armA-1ep-20260927-025236.log`  
**Training report:** `reports/M9a-REPORT-m9a-local-gpu1-armA-1ep-20260927-025236.md`

## Training config (context for test readout)

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

## Test harness protocol

Same canonical harness as Arm B: `training/eval_modernbert.py --subset test`,
**323** documents, **8192** max tokens, seed **42**, full sample (`--sample 0`).
**541** windows in `window_calibration`.

## Headline test metrics (eval harness — authoritative)

| Metric | Value |
| --- | ---: |
| doc_type_accuracy | **0.8638** (279/323) |
| subclass_accuracy_conditional | **0.4839** |
| window_calibration ECE | **0.0397** |
| window_calibration band ECE | 0.0574 |
| fast_path_rate | 0.0372 |

## Trainer vs eval harness (important)

The trainer exposes a lighter end-of-run test pass. **#112 and Hub test tables
must use the eval JSON below.**

| Surface | doc_type_accuracy | subclass (cond.) | window ECE |
| --- | ---: | ---: | ---: |
| Inline trainer `test_metrics.doc_type_acc` | **0.8669** | (see summary) | — |
| **Held-out eval harness** | **0.8638** | **0.4839** | **0.0397** |

The ~0.3 pp doc_type gap is expected: different code path and aggregation vs the
full `eval_modernbert.py` report. Always cite **0.8638** for gates and comparisons.

### Validation (epoch 1 only)

| Epoch | doc_type macro-F1 | doc_type doc acc | subclass objective | selection `gate_met` |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.8158 | 0.8528 | 0.238 | **false** |

Calibrated doc_type ECE **0.0524** > 0.05 on the sole epoch — no checkpoint
qualified under the lexicographic rule (selection epoch 0 placeholder).

## Cohorts (#104)

| cohort | n_docs | doc_type_accuracy | mean_agreement | window ECE |
| --- | ---: | ---: | ---: | ---: |
| single-window | 274 | 0.8723 | 1.0 | 0.1179 |
| multi-window | 49 | 0.8163 | 0.802 | 0.1024 |

Single-window cohort shows higher doc_type accuracy than multi-window; window ECE
is worse than Arm B on both cohorts (under-trained representation).

## Selective-risk sweep

**Refused** globally and per cohort:

```json
"reason": "no per-head ECE sidecar in the artifact (pre-#107 publish)"
```

This smoke was published with `--force-publish` before full selective-risk
sidecars were bundled; do not use Arm A for deployment threshold tuning.

## Per-head test macro-F1 (observed)

| head | macro-F1 |
| --- | ---: |
| insurance_claim | 0.7552 |
| merger_agreement | 0.16 |
| correspondence | 0.0944 |
| corporate_record | 0.0212 |
| contract | 0.0039 |

Support vectors match Arm B (same 323-doc test split); differences are purely
model quality.

## M9a #112 gates (held-out test)

Verified with `training/check_m9a_gates.py` → `gates-check-armA.txt`.

| Gate | Actual | Threshold | Status |
| --- | ---: | ---: | --- |
| contract test macro-F1 | 0.0039 | ≥ 0.20 | **NOT MET** |
| correspondence test macro-F1 | 0.0944 | ≥ 0.25 | **NOT MET** |
| doc_type test accuracy | 0.8638 | ≥ 0.89 | **NOT MET** |
| window ECE (doc_type calibrated) | 0.0397 | ≤ 0.05 | **MET** |

**Overall #112 gate verdict:** **FAIL** (3/4 gates not met — expected for
1-epoch λ_dt=0.5 compare arm).

### Report-only P0

| Gate | Actual | Threshold | met |
| --- | ---: | ---: | --- |
| P0 doc_type | 0.8638 | 0.95 | false |
| P0 subclass | 0.4839 | 0.75 | false |

## Hub release

| Field | Value |
| --- | --- |
| Model repo | `Lucius-Morningstar/mailroom-modernbert-classifier` |
| Release tag | `m9a-local-gpu1-armA-1ep-20260927-025236` |
| Eval on Hub | `eval_report_m9a-local-gpu1-armA-1ep-20260927-025236.json` |
| Publish mode | **`--force-publish`** (smoke compare; gates not required) |

## TEST-EVAL verdict

Arm A smoke confirms **λ_dt=0.5 + 1 epoch** under-trains relative to Arm B:

| Metric | Δ vs Arm B (B − A) |
| --- | ---: |
| doc_type_accuracy | **+0.0867** (B higher) |
| subclass_accuracy_conditional | **+0.0992** (B higher) |
| window ECE | −0.0138 (B better calibrated) |

Use for **ablation only**; do not promote to production default.
