# TEST-EVAL — held-out-plus (`m9a-local-20260927-014429`)

**Eval role:** extended monitoring pool (1,323 docs); **#112 gates remain on canonical 323 only** (`--subset test`).
**Run tag:** `m9a-local-20260927-014429`
**Eval JSON:** `eval_m9a-local-20260927-014429.json`
**Checkpoint:** `/checkpoints/latest`
**Artifact SHA:** `10cfcd60a91f5160e3870e1d0a666cc6436eee3062939962456ad65c77910caa`

## Harness protocol

| Parameter | Value |
| --- | --- |
| CLI | `training/eval_modernbert.py` |
| `--subset` | `heldout-plus` |
| Documents | **1323** (`--sample 0`) |
| Windows | **1554** |
| `--max-length` | 8192 |
| seed | 42 |

## Headline metrics (full 1,323)

| Metric | Value |
| --- | ---: |
| doc_type_accuracy | **0.0128** |
| subclass_accuracy_conditional | **0.2353** |
| window ECE | 0.1991 |
| fast_path_rate | 0.0 |

## Slices (canonical 323 vs Enron plus 1,000)

| slice | n | doc_type_acc | subclass (cond.) | corr. macro-F1 (cond.) |
| --- | ---: | ---: | ---: | ---: |
| canonical test | 323 | 0.0526 | 0.2353 | None |
| plus v1 (Enron) | 1000 | 0.0 | None | None |

### Per doc_type (plus slice)

| doc_type | n | doc_type_acc |
| --- | ---: | ---: |
| correspondence | 1000 | 0.0 |

## Per-head macro-F1 (full pool, harness)

- **contract**: macro_f1=None
- **corporate_record**: macro_f1=None
- **correspondence**: macro_f1=None
- **insurance_claim**: macro_f1=None
- **merger_agreement**: macro_f1=0.1571

## LLM sorter comparable metrics (`compare_runs` / shadow eval)

Structured block also embedded in eval JSON as `comparable_metrics` (schema `mailroom-ml/comparable-metrics/v1`). Pair with an LLM sorter eval export via `training/compare_runs.py` on shared filenames.

| Metric | Value |
| --- | ---: |
| doc_type_accuracy | 0.0128 |
| subclass_accuracy_conditional | 0.2353 |
| window_ece | 0.1991 |
| window_band_ece | 0.0 |
| mean_window_agreement | 1.0 |
| fast_path_rate | 0.0 |
| fast_path_n_docs | 0 |
| ood_rate | None |
| selective_risk | REFUSED: no per-head ECE sidecar in the artifact (pre-#107 publish) |
| latency_s_per_doc (Modal) | 0.183111 |

### Per-head macro-F1 (comparable block)

- **contract**: None
- **corporate_record**: None
- **correspondence**: None
- **insurance_claim**: None
- **merger_agreement**: 0.1571
