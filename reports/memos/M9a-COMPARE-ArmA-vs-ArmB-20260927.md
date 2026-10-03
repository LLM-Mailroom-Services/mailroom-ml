# M9a Arm A vs Arm B — comparison (2026-09-27)

Side-by-side summary of the GPU-1 smoke (Arm A) and the primary local train (Arm B). Full per-run write-ups:

- `reports/M9a-REPORT-m9a-local-20260927-014429.md` (Arm B)  
- `reports/M9a-REPORT-m9a-local-gpu1-armA-1ep-20260927-025236.md` (Arm A)

## Experimental arms

| | **Arm B (primary)** | **Arm A (smoke)** |
| --- | --- | --- |
| Run tag | `m9a-local-20260927-014429` | `m9a-local-gpu1-armA-1ep-20260927-025236` |
| GPU policy | GPU 0 | one-time GPU 1 exception |
| `loss_lambda_dt` | **0.65** | **0.5** |
| epochs | **3** | **1** |
| Eval JSON | `reports/eval_m9a-local-20260927-014429.json` | `reports/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json` |

## Held-out test (n=323)

| Metric | Arm B | Arm A | Δ (B − A) |
| --- | ---: | ---: | ---: |
| doc_type_accuracy | **0.9505** | 0.8638 | **+0.0867** |
| subclass_accuracy_conditional | **0.5831** | 0.4839 | **+0.0992** |
| window ECE | 0.0259 | 0.0397 | −0.0138 (better B) |

## Validation selection

| | Arm B | Arm A |
| --- | --- | --- |
| Selected epoch | 3 | 0 (none qualified) |
| Selection `gate_met` | **true** | false |
| Val doc_type macro-F1 | 0.9389 | 0.8158 (e1 only) |
| Val doc_type ECE (cal.) | 0.02 | 0.0524 (e1) |

## M9a #112 gates (held-out test)

| Gate | Arm B | Arm A |
| --- | --- | --- |
| contract macro-F1 ≥ 0.20 | NOT MET | NOT MET |
| correspondence macro-F1 ≥ 0.25 | NOT MET | NOT MET |
| doc_type acc ≥ 0.89 | **MET** | NOT MET |
| window ECE ≤ 0.05 | **MET** | **MET** |
| **Overall** | **FAIL (2/4)** | **FAIL (3/4)** |

## Interpretation

1. **λ_dt and depth:** Raising doc-type loss weight to 0.65 and training 3 epochs materially improves both routing accuracy and conditional subclass accuracy on the held-out set.  
2. **Production vs smoke:** Arm B is the candidate release; Arm A documents the cost of under-training for gate and λ ablation.  
3. **Remaining gap:** Both arms fail contract/correspondence subclass macro-F1 gates; Arm B additionally passes doc_type accuracy and calibration gates that Arm A misses.  
4. **Hub:** Both checkpoints are tagged on `Lucius-Morningstar/mailroom-modernbert-classifier` with distinct release tags and bundled `eval_report_<tag>.json` artifacts.

## Reference baselines

| doc_type test acc | source |
| ---: | --- |
| 0.8947 | run-3 (`reports/eval_run3_20260921.json`) |
| **0.9505** | **Arm B (this session primary)** |
| 0.8638 | Arm A smoke |
