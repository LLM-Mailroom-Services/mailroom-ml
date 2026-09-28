# TEST-EVAL — Arm B vs Arm A (2026-09-27)

**A:** `reports/eval_m9a-local-20260927-014429.json`  
**B:** `reports/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json`  

n_docs ArmB=323  ArmA=323  paired=323  pairing=document_id
bootstrap resamples=2000 seed=42

| metric | ArmB | ArmA | delta | 95% CI (A-B) |
|---|---:|---:|---:|---|
| doc_type_accuracy | 0.9505 | 0.8638 | 0.0867 | [0.0588, 0.1207] |
| window_ece | 0.0259 | 0.0397 | -0.0138 | — |
| fast_path_rate | 0.3034 | 0.0372 | 0.2662 | [0.2198, 0.3158] |
| selective_risk_threshold | 0.9000 | — | — | — |
| ood_rate | 0.0991 | 0.0619 | 0.0372 | — |

per-head macro-F1 (observed):
| head | ArmB | ArmA | delta |
|---|---:|---:|---:|
| contract | 0.0195 | 0.0039 | 0.0156 |
| corporate_record | 0.2485 | 0.0212 | 0.2273 |
| correspondence | 0.1289 | 0.0944 | 0.0345 |
| insurance_claim | 0.9889 | 0.7552 | 0.2337 |
| merger_agreement | 0.2381 | 0.1600 | 0.0781 |

## Reproduce

```bash
uv run python training/compare_runs.py --a reports/eval_m9a-local-20260927-014429.json --b reports/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json
```

## Artifacts

| `reports/eval_m9a-local-20260927-014429.json` | eval A |
| `reports/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json` | eval B |
