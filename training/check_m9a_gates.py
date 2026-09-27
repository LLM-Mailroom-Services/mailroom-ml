#!/usr/bin/env python3
"""Report #112 M9a success gates from an eval JSON (+ optional train summary).

Gates (mailroom-issues #112 / governance/M9a-HANDOFF.md):
  - contract test macro-F1 >= 0.20
  - correspondence test macro-F1 >= 0.25
  - doc_type test accuracy >= 0.89
  - window ECE (doc_type calibrated) <= 0.05

Also prints subclass_accuracy_conditional and per-head calibrated ECE so the
operator can see primary + subclass calibration together.

Exit 0 if all gates met, 1 otherwise. Report-only — does not mutate artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CONTRACT_F1 = 0.20
CORR_F1 = 0.25
DOC_TYPE_ACC = 0.89
WINDOW_ECE = 0.05


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        print(f"usage: {sys.argv[0]} <eval.json> [summary.json]", file=sys.stderr)
        return 2
    report = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    summary = {}
    if len(argv) >= 2 and Path(argv[1]).is_file():
        summary = json.loads(Path(argv[1]).read_text(encoding="utf-8"))

    per_head = report.get("per_head") or {}
    contract = _f((per_head.get("contract") or {}).get("macro_f1"))
    corr = _f((per_head.get("correspondence") or {}).get("macro_f1"))
    dt_acc = _f(report.get("doc_type_accuracy"))
    ece = _f((report.get("window_calibration") or {}).get("ece"))
    sc_acc = _f(report.get("subclass_accuracy_conditional"))

    checks = [
        ("contract_macro_f1", contract, CONTRACT_F1, "ge"),
        ("correspondence_macro_f1", corr, CORR_F1, "ge"),
        ("doc_type_accuracy", dt_acc, DOC_TYPE_ACC, "ge"),
        ("window_ece_calibrated", ece, WINDOW_ECE, "le"),
    ]
    print("=== M9a #112 gates ===")
    all_ok = True
    for name, actual, thr, how in checks:
        if actual is None:
            ok = False
            verdict = "MISSING"
        elif how == "ge":
            ok = actual >= thr
            verdict = "MET" if ok else "NOT MET"
        else:
            ok = actual <= thr
            verdict = "MET" if ok else "NOT MET"
        all_ok = all_ok and ok
        print(f"  {name}: actual={actual} threshold={thr} ({how}) -> {verdict}")

    print("=== calibration / subclass surfaces ===")
    print(f"  subclass_accuracy_conditional: {sc_acc}")
    wc = report.get("window_calibration") or {}
    print(f"  window_calibration: ece={wc.get('ece')} band_ece={wc.get('band_ece')}")
    for head, info in sorted(per_head.items()):
        if not isinstance(info, dict):
            continue
        print(
            f"  per_head.{head}: macro_f1={info.get('macro_f1')} "
            f"support={info.get('support')} "
            f"ece_calibrated={info.get('ece_calibrated')}"
        )

    sel = (summary.get("checkpoint_selection") or {}) if summary else {}
    if sel:
        print("=== trainer selection (from summary.json) ===")
        print(
            f"  epoch={sel.get('epoch')} gate_met={sel.get('gate_met')} "
            f"subclass_obj={sel.get('subclass_objective')} "
            f"doc_type_macro_f1={sel.get('macro_f1')} ece={sel.get('ece')}"
        )
        excl = (sel.get("head_exclusion_policy") or {}).get("excluded") or {}
        if excl:
            print("  head_exclusion_policy:")
            for h, meta in sorted(excl.items()):
                print(f"    {h}: {meta}")

    print("=== overall ===")
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
