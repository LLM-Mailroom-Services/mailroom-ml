# PROBE — corr-head un-exclusion ceiling experiment (`m9c-20261014-113614`)

**Question:** would un-excluding the correspondence head unlock fast-path skipping on real (Enron-style) mail, without retraining?
**Method:** twin eval of the 1,323-doc heldout-plus pool on a volume probe copy (`runs/probe-corr-unex`) of the released M9c bundle with a single policy flip in `summary.json` (`head_exclusion_policy.excluded.correspondence: true → false`). Same seed/pool as the committed receipt; bytes otherwise identical (probe safetensors sha == released artifact sha `05301a83…`); fp32 ONNX serving-identical eval path (`load_bundle` prefers `model.onnx`). Cost **$0.0554** (wall 256 s, L4).

## Result

| read | published policy | probe (corr un-excluded) |
|---|---:|---:|
| fast_path_rate | 0.1005 | **0.1081** |
| corr-gt docs reaching fast path | 0% (policy) | **10 / 1,085 = 0.9%** |
| sc_correct among skips | — | **1.000** |
| sc_correct among gate-rejected corr docs | — | 0.7907 |
| dt accuracy / subclass_cond / corr F1 | 0.9766 / 0.8011 / 0.3174 | identical (routing-only flip) |

## Read-out

Corr-head **coverage at the production gate** is the binding constraint under mail shift — not the `#107` exclusion policy. Un-exclusion (or post-hoc calibration alone) buys ~nothing; only higher calibrated confidence on mail-family mass — i.e. the adoption arm's training — moves it. This is the measured justification for the #246 arm; its objective should target calibrated confidence/skip coverage under shift, not macro-F1 alone. The head-exclusion policy stays correct as shipped on the released artifact.

## Reproduce

```bash
# probe copy with the single policy flip, then volume put → runs/probe-corr-unex
uv run python - <<'PY'  # flip inside artifacts/probe-corr-unex/summary.json
…
PY
uv run --extra deploy modal volume put modernbert-checkpoints artifacts/probe-corr-unex runs/probe-corr-unex
uv run --extra deploy modal run deploy/eval_app.py --module runs/probe-corr-unex \
  --subset heldout-plus --sample 0 --seed 42 --as-json \
  --out /private/tmp/opencode/m9c_probe_corrunex.json
```

Probe copy: `artifacts/probe-corr-unex/` (local, untracked). Eval JSON: `m9c_probe_corrunex.json` (local). Cross-links: issue `mailroom-issues#246` (comment 6171538652), released tag `m9c-20261014-113614`.
