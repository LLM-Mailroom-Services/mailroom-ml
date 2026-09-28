# mailroom-ml reports

Human-readable experiment logs and eval write-ups for the ModernBERT ingest
classifier. **Every metric in generated markdown is computed from recorded
artifacts** (`eval_*.json`, optional `summary.json`, telemetry sidecars) via
`training/write_eval_report.py` — not hand-edited.

## Generator

```bash
uv run python training/write_eval_report.py test --run-tag <tag> \
  --eval-json reports/eval_<tag>.json \
  --summary-json data/modernbert_training/runs/<tag>/latest/summary.json

uv run python training/write_eval_report.py heldout-plus --run-tag <tag> \
  --eval-json reports/eval_<tag>-heldout-plus.json \
  --baseline-json reports/eval_<tag>.json

uv run python training/write_eval_report.py compare --a ... --b ... \
  --out reports/TEST-EVAL/TEST-EVAL-COMPARE-....md
```

Post-train: `./training/complete_run.sh --run-tag <tag>` runs eval, gates, and
markdown generation when eval JSON is present.

Sandbox mirror: `./training/sync_reports_to_sandbox.sh --sandbox-root ../mailroom-sandbox \
  --run-tag <tag> --run-number <nn>`.

## Report kinds

| Kind | ID | Path pattern | Required sections (order) |
| --- | --- | --- | --- |
| Held-out test (323) | `test_eval_323` | `reports/TEST-EVAL/TEST-EVAL-REPORT-<run_tag>.md` | metadata → training config (if summary) → harness protocol → headline metrics → trainer vs harness → cohorts → selective-risk → per-head macro-F1 → head ECE policy → #112 gates → P0 gates → baseline compare → Hub block → analyst insights → per-doc (bounded) → verdict → reproduce → artifacts |
| Held-out-plus (1,323) | `test_eval_heldout_plus` | `reports/TEST-EVAL/TEST-EVAL-REPORT-heldout-plus-<run_tag>.md` | metadata → harness → headline (full pool) → 323 vs plus slices → per-head → comparable metrics → insights → per-doc errors-only → reproduce → artifacts |
| Training run | `training_run` | `reports/M9a-REPORT-<run_tag>.md` | metadata → config → validation epochs → selection → exclusion policy → held-out test summary → verdict |
| Compare | `compare_runs` | `reports/TEST-EVAL/TEST-EVAL-COMPARE-*.md` | preamble → paired bootstrap table → per-head delta → insights → reproduce → artifacts |
| Dataset spec | `dataset_spec` | `reports/HOLDOUT-PLUS-V1-REPORT.md` | Manual dataset lineage (not regenerated from eval JSON) |

Section order follows the mailroom-sandbox SAND-032 report contract (metadata,
headline, serving/telemetry when present, observations, insights, method/gates,
strata, per-doc, reproduce, artifacts).

## Canonical JSON

| Artifact | Role |
| --- | --- |
| `reports/eval_<run_tag>.json` | 323-doc held-out test (authoritative for #112) |
| `reports/eval_<run_tag>-heldout-plus.json` | Extended 1,323-doc monitoring pool |
| `reports/TEST-EVAL/` | TEST-EVAL copies, compare docs, `MANIFEST.json`, gates-check txt |
| `reports/preflight_latest.json` | Operator QA (JSON only) |

## Index

- [TEST-EVAL harness index](./TEST-EVAL/README.md)
- [Held-out-plus v1 dataset spec](./HOLDOUT-PLUS-V1-REPORT.md)
- M9a gates: `training/check_m9a_gates.py` / `governance/M9a-HANDOFF.md`
