# TEST-EVAL — Arm A vs Arm B comparison (2026-09-27)

Side-by-side **held-out test harness** results for the M9a local session. Per-run
TEST-EVAL depth:

- [TEST-EVAL-REPORT-m9a-local-20260927-014429.md](./TEST-EVAL-REPORT-m9a-local-20260927-014429.md) (Arm B primary)  
- [TEST-EVAL-REPORT-m9a-local-gpu1-armA-1ep-20260927-025236.md](./TEST-EVAL-REPORT-m9a-local-gpu1-armA-1ep-20260927-025236.md) (Arm A smoke)

Machine-readable index: [MANIFEST.json](./MANIFEST.json).

## Experimental arms

| | **Arm B (primary)** | **Arm A (smoke)** |
| --- | --- | --- |
| Run tag | `m9a-local-20260927-014429` | `m9a-local-gpu1-armA-1ep-20260927-025236` |
| GPU policy | GPU 0 | one-time GPU 1 exception |
| `loss_lambda_dt` | **0.65** | **0.5** |
| epochs | **3** | **1** |
| TEST-EVAL JSON | `eval_m9a-local-20260927-014429.json` | `eval_m9a-local-gpu1-armA-1ep-20260927-025236.json` |
| Production intent | **yes** | ablation only |

## Held-out test harness (n=323, identical protocol)

Both rows below are from `training/eval_modernbert.py --subset test`, 8192 tokens,
seed 42 — **not** trainer inline test alone.

| Metric | Arm B | Arm A | Δ (B − A) |
| --- | ---: | ---: | ---: |
| doc_type_accuracy | **0.9505** | 0.8638 | **+0.0867** |
| subclass_accuracy_conditional | **0.5831** | 0.4839 | **+0.0992** |
| window ECE | 0.0259 | 0.0397 | −0.0138 (B better) |
| band ECE | 0.0525 | 0.0574 | −0.0049 (B better) |
| fast_path_rate | 0.3034 | 0.0372 | +0.2662 |

### Trainer vs harness (Arm A reconciliation)

| Surface | Arm A doc_type |
| --- | ---: |
| Trainer `test_metrics.doc_type_acc` | 0.8669 |
| **Eval harness (use for gates)** | **0.8638** |

Arm B: cite harness **0.9505**; validation epoch-3 doc acc was 0.9331 (trainer).

## Cohorts (#104)

| cohort | Arm B doc_type acc | Arm A doc_type acc |
| --- | ---: | ---: |
| single-window (n=274) | 0.9489 | 0.8723 |
| multi-window (n=49) | 0.9592 | 0.8163 |

Arm B improves both cohorts; Arm A multi-window drops to 0.8163 with lower mean
window agreement (0.802).

## Validation selection (trainer — not #112 gates)

| | Arm B | Arm A |
| --- | --- | --- |
| Selected epoch | 3 | 0 (none qualified) |
| Selection `gate_met` | **true** | false |
| Val doc_type macro-F1 | 0.9389 | 0.8158 (e1 only) |
| Val doc_type ECE (cal.) | 0.02 | 0.0524 (e1) |

Strong test numbers on Arm B align with a checkpoint that passed validation
calibration gates at epoch 3.

## Per-head test macro-F1

| head | Arm B | Arm A |
| --- | ---: | ---: |
| insurance_claim | **0.9889** | 0.7552 |
| corporate_record | **0.2485** | 0.0212 |
| merger_agreement | **0.2381** | 0.16 |
| correspondence | **0.1289** | 0.0944 |
| contract | **0.0195** | 0.0039 |

Both arms fail #112 on contract and correspondence; Arm B lifts all heads vs
smoke, with insurance_claim near ceiling.

## M9a #112 gates (held-out test)

| Gate | Arm B | Arm A |
| --- | --- | --- |
| contract macro-F1 ≥ 0.20 | NOT MET | NOT MET |
| correspondence macro-F1 ≥ 0.25 | NOT MET | NOT MET |
| doc_type acc ≥ 0.89 | **MET** | NOT MET |
| window ECE ≤ 0.05 | **MET** | **MET** |
| **Overall** | **FAIL (2/4)** | **FAIL (3/4)** |

Gate checker outputs: [gates-check-armB.txt](./gates-check-armB.txt),
[gates-check-armA.txt](./gates-check-armA.txt).

## Interpretation

1. **λ_dt and depth:** Raising doc-type loss weight to 0.65 and training 3 epochs
   materially improves routing and conditional subclass accuracy on the fixed
   323-doc harness (+8.7 pp / +9.9 pp vs smoke).
2. **Production vs smoke:** Arm B is the candidate release; Arm A documents the
   cost of under-training for λ and epoch ablation.
3. **Remaining gap:** Both arms fail contract/correspondence subclass macro-F1
   gates; Arm B additionally passes doc_type accuracy and calibration gates that
   Arm A misses.
4. **Hub:** Both checkpoints are tagged on
   `Lucius-Morningstar/mailroom-modernbert-classifier` with distinct release tags
   and bundled `eval_report_<tag>.json` artifacts (copies in this directory).

## Reference baselines (doc_type test accuracy, harness)

| doc_type test acc | source |
| ---: | --- |
| 0.8947 | run-3 (`reports/eval_run3_20260921.json`) |
| **0.9505** | **Arm B (session primary)** |
| 0.8638 | Arm A smoke |

Arm B improves **+5.58 pp** vs run-3 on doc_type; conditional subclass also
rises (0.5831 vs 0.5260).
