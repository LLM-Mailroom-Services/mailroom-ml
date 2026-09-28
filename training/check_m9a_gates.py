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

from mailroom_ml.m9a_gates import gate_status, gates_all_met, m9a_gate_rows

_KEY = {
    "contract test macro-F1": "contract_macro_f1",
    "correspondence test macro-F1": "correspondence_macro_f1",
    "doc_type test accuracy": "doc_type_accuracy",
    "window ECE (doc_type calibrated)": "window_ece_calibrated",
}


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        print(f"usage: {sys.argv[0]} <eval.json> [summary.json]", file=sys.stderr)
        return 2
    report = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    summary = {}
    if len(argv) >= 2 and Path(argv[1]).is_file():
        summary = json.loads(Path(argv[1]).read_text(encoding="utf-8"))

    print("=== M9a #112 gates ===")
    for name, actual, thr, how in m9a_gate_rows(report):
        st = gate_status(actual, thr, how)
        key = _KEY.get(name, name)
        print(f"  {key}: actual={actual} threshold={thr} ({how}) -> {st}")

    print("=== calibration / subclass surfaces ===")
    print(f"  subclass_accuracy_conditional: {report.get('subclass_accuracy_conditional')}")
    wc = report.get("window_calibration") or {}
    print(f"  window_calibration: ece={wc.get('ece')} band_ece={wc.get('band_ece')}")
    per_head = report.get("per_head") or {}
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
    print("PASS" if gates_all_met(report) else "FAIL")
    return 0 if gates_all_met(report) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
