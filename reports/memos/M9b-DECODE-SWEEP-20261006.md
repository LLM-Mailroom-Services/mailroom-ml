# M9b decode-residual sweep — 2026-10-06

**Checkpoint:** `20261006-021245` (`/checkpoints/runs/20261006-021245`, local copy under `data/modernbert_training/runs/20261006-021245/`).  
**Pool:** same 323-doc `test` split as last night (`sample=0`). No train. No Hub publish.  
**Method:** `--subclass-decode-logit-adjust d` on raw subclass logits (`tau_eff = 1.0 − d`). Default `d=0` is last night’s argmax (not re-run).  
**GPU:** one Modal L4 (`--profile=exios66`, profile not activated). Four `d` values in **one** container. Remote wall **696 s** (~11.6 min). Est. **~$0.15** (telemetry `gpu_est=$0.154733` on the sweep wall; **not** 4× that). Live GPU tasks after: **0**.

**Verdict: FAIL.** No `d` meets both #112 gates (contract ≥ 0.20 **and** correspondence ≥ 0.25). Closest contract is **d=0.5** at **0.1697**. Do not Hub-publish. Next lever is **#113 CUAD**, not another tau train.

## Gates vs `d`

| `d` | `τ_eff` | n | contract F1 (≥0.20) | correspondence F1 (≥0.25) | both | doc_type acc | window ECE | merger F1 | service rec. (dt-correct) | supply rec. (dt-correct) |
| ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 0 (baseline) | 1.0 | 323 | **0.162** NOT | **0.351** MET | no | 0.9505 | 0.0152 | 0.000 | 0/7 | 0/4 |
| 0.25 | 0.75 | 323 | **0.1477** NOT | **0.4394** MET | no | 0.9505 | 0.0152 | 0.152 | 0/7 | 0/4 |
| **0.5** | 0.5 | 323 | **0.1697** NOT | **0.4254** MET | no | 0.9505 | 0.0152 | 0.160 | 0/7 | 0/4 |
| 0.75 | 0.25 | 323 | **0.1431** NOT | **0.4052** MET | no | 0.9505 | 0.0152 | 0.160 | 0/7 | 0/4 |
| 1.0 | 0.0 | 323 | **0.1159** NOT | **0.4143** MET | no | 0.9505 | 0.0152 | 0.160 | 0/7 | 0/4 |

JSON: `reports/json/eval_20261006-021245-d{0.25,0.5,0.75,1.0}.json`. Baseline `reports/json/eval_20261006-021245.json`.

GT support is 8 service / 5 supply (one of each is a doc_type miss, so conditional n is 7 / 4). Recalled **zero** service and **zero** supply at every `d`.

## What moved (and what did not)

- **Doc-type acc and ECE are invariant** — decode adjust is subclass-only.
- **Correspondence stays well above 0.25** and *improves*: attorney_demand on dt-correct correspondence drops **39 → 5 → 0** by `d≥0.5`.
- **Merger recovers from collapse:** 17/17 `mixed_cash_stock_election` at `d=0` → **0** election preds for `d≥0.25`; merger macro-F1 **0 → ~0.16**. Not a #112 gate.
- **Contract does not clear 0.20.** Correct count stays ~11–14 / 53. Peak F1 is `d=0.5` (0.1697), still **0.030** short. Service/supply never leave F1=0; mass stays on co_branding / transportation / license / maintenance, not on the missing CUAD families.

## Operator follow-up

- **Do not** retrain `tau=1.0` or add epochs on this run.
- **Do not** Hub-publish a decode `d` (none won).
- **#113 CUAD** (data-publish on service / supply / ip / maintenance / sponsorship) is the next authorized lever if the operator wants another train after a new data pin.
