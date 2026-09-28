"""M9a #112 held-out test gate thresholds (mailroom-issues #112)."""

from __future__ import annotations

CONTRACT_F1 = 0.20
CORR_F1 = 0.25
DOC_TYPE_ACC = 0.89
WINDOW_ECE = 0.05


def _f(x: object) -> float | None:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def m9a_gate_rows(report: dict) -> list[tuple[str, float | None, float, str]]:
    """Display name, actual, threshold, comparator ('ge' or 'le')."""
    per_head = report.get("per_head") or {}
    contract = _f((per_head.get("contract") or {}).get("macro_f1"))
    corr = _f((per_head.get("correspondence") or {}).get("macro_f1"))
    dt_acc = _f(report.get("doc_type_accuracy"))
    ece = _f((report.get("window_calibration") or {}).get("ece"))
    return [
        ("contract test macro-F1", contract, CONTRACT_F1, "ge"),
        ("correspondence test macro-F1", corr, CORR_F1, "ge"),
        ("doc_type test accuracy", dt_acc, DOC_TYPE_ACC, "ge"),
        ("window ECE (doc_type calibrated)", ece, WINDOW_ECE, "le"),
    ]


def gate_status(actual: float | None, thr: float, how: str) -> str:
    if actual is None:
        return "MISSING"
    if how == "ge":
        return "MET" if actual >= thr else "NOT MET"
    return "MET" if actual <= thr else "NOT MET"


def gates_all_met(report: dict) -> bool:
    for _name, actual, thr, how in m9a_gate_rows(report):
        if actual is None:
            return False
        if how == "ge" and actual < thr:
            return False
        if how == "le" and actual > thr:
            return False
    return True


def format_gates_cli(report: dict) -> str:
    """Plain-text block matching ``check_m9a_gates.py`` stdout."""
    lines = ["=== M9a #112 gates ==="]
    all_ok = True
    key_map = {
        "contract test macro-F1": "contract_macro_f1",
        "correspondence test macro-F1": "correspondence_macro_f1",
        "doc_type test accuracy": "doc_type_accuracy",
        "window ECE (doc_type calibrated)": "window_ece_calibrated",
    }
    for name, actual, thr, how in m9a_gate_rows(report):
        st = gate_status(actual, thr, how)
        ok = st == "MET"
        all_ok = all_ok and ok
        key = key_map.get(name, name)
        lines.append(f"  {key}: actual={actual} threshold={thr} ({how}) -> {st}")
    lines.append("=== overall ===")
    lines.append("PASS" if all_ok else "FAIL")
    return "\n".join(lines)
