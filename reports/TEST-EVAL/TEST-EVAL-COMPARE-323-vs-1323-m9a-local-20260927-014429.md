# TEST-EVAL — 323 vs 1,323 (`m9a-local-20260927-014429`)

Paired bootstrap uses **323 overlapping filenames** only (canonical test block).

# compare_runs: canonical323 vs heldout1323

n_docs canonical323=323  heldout1323=1323  paired=323  pairing=document_id
bootstrap resamples=2000 seed=42

| metric | canonical323 | heldout1323 | delta | 95% CI (A-B) |
|---|---:|---:|---:|---|
| doc_type_accuracy | 0.9505 | 0.0128 | 0.9377 | [0.8607, 0.9288] |
| window_ece | 0.0259 | 0.1991 | -0.1732 | — |
| fast_path_rate | 0.3034 | 0.0000 | 0.3034 | [0.2539, 0.3560] |
| selective_risk_threshold | 0.9000 | — | — | — |
| ood_rate | 0.0991 | — | — | — |

per-head macro-F1 (observed):
| head | canonical323 | heldout1323 | delta |
|---|---:|---:|---:|
| contract | 0.0195 | — | — |
| corporate_record | 0.2485 | — | — |
| correspondence | 0.1289 | — | — |
| insurance_claim | 0.9889 | — | — |
| merger_agreement | 0.2381 | 0.1571 | 0.0810 |

