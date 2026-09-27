# TEST-EVAL — M9a held-out test harness reports (2026-09-27)

This directory consolidates **authoritative held-out test evaluations** for the
M9a local training session. Metrics here come from `training/eval_modernbert.py`
(the GPU/CPU eval harness), not from inline trainer `test_metrics` alone.

## Index

| Document | Description |
| --- | --- |
| [MANIFEST.json](./MANIFEST.json) | Machine-readable run list, artifact SHA, #112 gate snapshot |
| [TEST-EVAL-REPORT-m9a-local-20260927-014429.md](./TEST-EVAL-REPORT-m9a-local-20260927-014429.md) | **Arm B primary** (3 epochs, λ_dt=0.65) — full test harness report |
| [TEST-EVAL-REPORT-m9a-local-gpu1-armA-1ep-20260927-025236.md](./TEST-EVAL-REPORT-m9a-local-gpu1-armA-1ep-20260927-025236.md) | **Arm A smoke** (1 epoch, λ_dt=0.5) — full test harness report |
| [TEST-EVAL-COMPARE-ArmA-vs-ArmB-20260927.md](./TEST-EVAL-COMPARE-ArmA-vs-ArmB-20260927.md) | Side-by-side comparison and interpretation |
| [gates-check-armB.txt](./gates-check-armB.txt) | `check_m9a_gates.py` output (Arm B) |
| [gates-check-armA.txt](./gates-check-armA.txt) | `check_m9a_gates.py` output (Arm A) |
| [eval_m9a-local-20260927-014429.json](./eval_m9a-local-20260927-014429.json) | Eval artifact copy (constellation / import friendly) |
| [eval_m9a-local-gpu1-armA-1ep-20260927-025236.json](./eval_m9a-local-gpu1-armA-1ep-20260927-025236.json) | Eval artifact copy |

**Sibling training-focused write-ups** (validation epochs, wall time, Hub publish
notes) remain under `reports/M9a-REPORT-*.md` and
`reports/M9a-COMPARE-ArmA-vs-ArmB-20260927.md`. This folder adds explicit
TEST-EVAL framing: harness protocol, trainer vs eval reconciliation, cohorts,
and #112 gates.

## Held-out test harness (canonical 323 docs)

| Property | Value |
| --- | --- |
| CLI | `training/eval_modernbert.py` |
| Subset | `--subset test` (default) |
| Corpus | Staged tree `data/modernbert_training/stage` (`split == test`) |
| n_docs | **323** (full split; `--sample 0`) |
| max_length | **8192** tokens per window |
| seed | 42 |
| Never used for | training, validation checkpoint selection, or threshold tuning on this split |

The harness emits JSON with: doc_type and conditional subclass accuracy,
window-level calibrated ECE (+ band ECE), per-head test macro-F1 with per-class
**support**, #104 single- vs multi-window **cohorts**, selective-risk sweep
(when sidecars permit), and **recorded_gates** (P0 report-only thresholds
separate from #112).

Reproduce gate summary:

```bash
python3.11 training/check_m9a_gates.py reports/TEST-EVAL/eval_m9a-local-20260927-014429.json
```

## M9a #112 success gates (held-out test)

From `governance/M9a-HANDOFF.md` / mailroom-issues **#112**:

| Gate | Threshold |
| --- | ---: |
| contract test macro-F1 | ≥ 0.20 |
| correspondence test macro-F1 | ≥ 0.25 |
| doc_type test accuracy | ≥ 0.89 |
| window ECE (doc_type calibrated) | ≤ 0.05 |

**Session verdict:** both evaluated checkpoints **FAIL** overall #112 (subclass
heads); Arm B **passes** doc_type accuracy and calibration; Arm A smoke passes
calibration only.

## Trainer vs eval harness

| Run | Trainer / inline test (reference) | Eval harness (authoritative for #112) |
| --- | ---: | ---: |
| Arm B | See `summary.json` → `test_metrics` when present | doc_type **0.9505**, subclass cond. **0.5831** |
| Arm A smoke | `test_metrics.doc_type_acc` **0.8669** (lighter in-train pass) | doc_type **0.8638**, subclass cond. **0.4839** |

Always cite **`eval_*.json`** / Hub `eval_report_<tag>.json` for release gates
and model-card test tables.

## Related artifacts

- Hub narrative: `docs/mailroom-modernbert-classifier-model-card.md`
- Run-3 baseline eval: `reports/eval_run3_20260921.json` (doc_type acc 0.8947)
- Extended test pool (not used for these two reports): `reports/HOLDOUT-PLUS-V1-REPORT.md` (1,323 docs via `--subset heldout-plus`)
