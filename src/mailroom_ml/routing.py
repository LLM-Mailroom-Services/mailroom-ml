"""Intake routing + PASS/FAIL gate layer (mailroom-issues #85 M1/M2/M5/#86-#91).

Implements the constellation intake-overhaul contract verbatim:

- ``should_llm_intake`` — the epic gate precedence (plan §9): no text ->
  False; messy -> True; ml_triage missing/failed -> True; route != fast_path
  -> True; fast_path -> skip ONLY in ``mode == "skip"`` with an allowlisted
  gate PASS (#102) — shadow/verify always run the sorter.
  ``MAILROOM_BERT_INTAKE=0`` preserves today's behavior exactly (flag-off is
  the instant rollback, #85 M6).
- ``should_bert_intake`` — M2 gate: context-fit (chars <= max, tokens <=
  8,192), not messy, model available, flag on; machine-readable ``reason``
  for observability (#87).
- ``build_handoff`` — M1 ``intake_handoff`` schema v1, the exact JSON shape
  the epic spells out: schema_version, method, routing, triage, sections,
  quality, gate, provenance (#86).
- ``evaluate_intake_gate`` — M5 PASS/FAIL wiring implementing P1-P7 / F1-F7
  as a documented named-check table; reviewer stays blind (agreement is
  computed here, never surfaced to the reviewer); skip eligibility only when
  ALL PASS criteria hold (#90).

Fail-open invariant (plan D10, #85): any malformed input / missing model /
exception degrades to the LLM path; ``evaluate_intake_gate`` never raises on
a broken handoff — it returns FAIL with the failure recorded.
"""
from __future__ import annotations

from typing import Any

from mailroom_ml.config import (
    BERT_INTAKE_MAX_CHARS,
    BERT_INTAKE_MIN_CONFIDENCE,
    BERT_INTAKE_MODE,
    GATE_ALLOWLISTED_START,
    INTAKE_HANDOFF_SCHEMA_VERSION,
    MAILROOM_BERT_INTAKE_DEFAULT,
    MAX_TOKENS,
)

__all__ = [
    "should_llm_intake",
    "should_bert_intake",
    "context_fit",
    "build_handoff",
    "evaluate_intake_gate",
    "PASS_CHECKS",
    "FAIL_CHECKS",
]


# ---------------------------------------------------------------------------
# Routing gates (epic Phase 1 + M2 #87)
# ---------------------------------------------------------------------------

def context_fit(text: str, *, max_chars: int = BERT_INTAKE_MAX_CHARS,
                token_estimate: int | None = None,
                max_tokens: int = MAX_TOKENS) -> bool:
    """M2 context-fit: chars budget AND (when supplied) token budget.

    Char budget mirrors the epic's ``bert_intake_max_chars`` (30,000 chars
    ≈ 8,192 tokens at ``CHARS_PER_TOKEN``); token estimate comes from the
    pipeline stats (``estimate_tokens``) — true tokenizer
    counts are enforced by the inference layer at encode time (never
    truncate; oversize -> LLM path).
    """
    if len(text) > max_chars:
        return False
    if token_estimate is not None and token_estimate > max_tokens:
        return False
    return True


def should_bert_intake(text: str, stats: dict[str, Any] | None = None, *,
                       flag: int | None = None,
                       mode: str | None = None,
                       model_available: bool = True,
                       max_chars: int = BERT_INTAKE_MAX_CHARS,
                       max_tokens: int = MAX_TOKENS) -> tuple[bool, str]:
    """M2 routing decision: run the ModernBERT path?  Returns (bool, reason).

    False reasons (machine-readable, #87): ``flag_off`` (instant rollback,
    today's behavior), ``empty`` (no text), ``messy`` (clerk's flag routes
    first — the LLM cleans), ``oversize_chars`` / ``oversize_tokens``
    (context-fit fails), ``no_model`` (artifact unavailable — fail-open).
    ``mode`` (shadow|verify|skip) never blocks BERT compute — all three
    calculate the triage; the MODE constrains the GATE (see
    ``evaluate_intake_gate``), not the decision to compute.
    """
    if flag is None:
        flag = int(_env_int("MAILROOM_BERT_INTAKE", MAILROOM_BERT_INTAKE_DEFAULT))
    if mode is None:
        mode = _env_str("BERT_INTAKE_MODE", BERT_INTAKE_MODE)
    if not flag:
        return False, "flag_off"
    if not text.strip():
        return False, "empty"
    stats = stats or {}
    if stats.get("messy"):
        return False, "messy"
    if not model_available:
        return False, "no_model"
    tok = stats.get("token_estimate")
    if not context_fit(text, max_chars=max_chars, token_estimate=tok,
                       max_tokens=max_tokens):
        return False, "oversize_tokens" if tok and tok > max_tokens \
            else "oversize_chars"
    return True, "within_bert_ctx"


def should_llm_intake(text: str, stats: dict[str, Any] | None = None,
                      ml_triage: dict[str, Any] | None = None, *,
                      flag: int | None = None,
                      mode: str | None = None,
                      gate: dict[str, Any] | None = None,
                      max_chars: int = BERT_INTAKE_MAX_CHARS,
                      max_tokens: int = MAX_TOKENS) -> bool:
    """Epic gate precedence (plan §9): when must the LLM path run?

    Order: empty text -> False (nothing to do); ``stats.messy`` -> True;
    BERT flag off -> True IF oversize (today's behavior preserved: messy or
    > max chars), else False; ml_triage missing/failed/exception -> True
    (fail-open, plan D10); route != fast_path -> True.

    Flag on + ``route == "fast_path"``: the sorter is skipped ONLY in
    ``mode == "skip"`` with an allowlisted gate PASS (#102). Shadow and verify
    never skip — the mode constrains the GATE, not the decision to compute
    (epic ladder: shadow = sorter always runs; verify = PASS requires
    agreement; only skip may bypass, allowlisted + PASS). A missing ``gate``
    therefore means "run the sorter" (the safe default).

    ``stats`` mirrors the pipeline's ``looks_messy`` + char stats:
    ``{"messy": bool, "chars": int, "token_estimate": int}`` — the caller
    (the governed intake node) supplies what it has; None stats are treated
    as clean-but-unknown (messy asserts only when explicitly true).
    """
    if flag is None:
        flag = int(_env_int("MAILROOM_BERT_INTAKE", MAILROOM_BERT_INTAKE_DEFAULT))
    if not text.strip():
        return False
    stats = stats or {}
    if stats.get("messy"):
        return True
    oversize = len(text) > max_chars or (
        stats.get("token_estimate") or 0) > max_tokens
    if not flag:
        # today's behavior: LLM intake only for messy/oversize (pre-BERT)
        return bool(oversize)
    if ml_triage is None or ml_triage.get("status") != "ok" \
            or ml_triage.get("route") != "fast_path":
        return True
    # flag on + fast_path: only skip-mode + allowlisted + gate PASS may skip
    if mode is None:
        mode = _env_str("BERT_INTAKE_MODE", BERT_INTAKE_MODE)
    if mode != "skip":
        return True
    if gate is None or not gate.get("eligible_for_sorter_skip"):
        return True
    if ml_triage.get("doc_type") not in GATE_ALLOWLISTED_START:
        return True
    return False


# ---------------------------------------------------------------------------
# M1 intake_handoff schema v1 (#86)
# ---------------------------------------------------------------------------

def build_handoff(
    *,
    method: str = "deterministic",
    routing_path: str = "clerk_only",
    routing_reason: str = "flag_off",
    doc_chars: int = 0,
    doc_token_estimate: int = 0,
    doc_type: str | None = None,
    doc_subclass: str | None = None,
    confidence: float = 0.0,
    gist: str = "",
    keywords: list[str] | None = None,
    sections: list[dict[str, Any]] | None = None,
    quality: dict[str, Any] | None = None,
    guard_failures: list[str] | None = None,
    eligible_for_sorter_skip: bool = False,
    recommended_action: str = "force_llm_sorter",
    bert_threshold: float = BERT_INTAKE_MIN_CONFIDENCE,
    notes: str = "",
    model_id: str | None = None,
    model_revision: str | None = None,
    intake_windows: int = 0,
    changes_applied: list[str] | None = None,
) -> dict[str, Any]:
    """The M1 ``intake_handoff`` dict — schema v1, exact epic shape (#86).

    Defaults preserve today's behavior: flag off -> clerk-only routing,
    no triage claim, not skip-eligible, force the LLM sorter.  Any caller
    builds the handoff from the actual runtime; the shape is the contract.
    """
    quality = quality or {}
    sections = sections or []
    return {
        "schema_version": INTAKE_HANDOFF_SCHEMA_VERSION,
        "method": method,
        "routing": {
            "path": routing_path,
            "reason": routing_reason,
            "bert_context_limit_tokens": MAX_TOKENS,
            "doc_chars": doc_chars,
            "doc_token_estimate": doc_token_estimate,
        },
        "triage": {
            "primary_doc_class": doc_type,
            "doc_subclass": doc_subclass,
            "confidence": confidence,
            "gist": gist,
            "keywords": keywords or [],
        },
        "sections": sections,
        "quality": {
            "messy": bool(quality.get("messy", False)),
            "context_fit": bool(quality.get("context_fit", False)),
            "coverage": float(quality.get("coverage", 0.0)),
            "section_map_ok": bool(quality.get("section_map_ok", True)),
            "triage_vocab_ok": bool(quality.get("triage_vocab_ok", False)),
            "guard_failures": guard_failures or [],
        },
        "gate": {
            "eligible_for_sorter_skip": eligible_for_sorter_skip,
            "recommended_action": recommended_action,
            "bert_threshold": bert_threshold,
            "notes": notes,
        },
        "provenance": {
            "model_id": model_id,
            "model_revision": model_revision,
            "intake_windows": intake_windows,
            "changes_applied": changes_applied or [],
        },
    }


# ---------------------------------------------------------------------------
# M5 PASS/FAIL gate (#90) — P1-P7 / F1-F7
# ---------------------------------------------------------------------------

PASS_CHECKS: dict[str, str] = {
    "P1": "method == bert, quality.context_fit and coverage == 1.0",
    "P2": "quality.messy == false and guard_failures == []",
    "P3": "triage.confidence >= bert_intake_min_confidence (calibrated)",
    "P4": "triage_vocab_ok — class/subclass in live taxonomy",
    "P5": "section_map_ok (or sections empty-by-design for short forms)",
    "P6": "sorter agreement in verify mode (or skip authorized by config)",
    "P7": "reviewer agreement, when the reviewer fired (blind)",
}

FAIL_CHECKS: dict[str, str] = {
    "F1": "oversize / context_fit == false",
    "F2": "messy == true",
    "F3": "confidence below threshold",
    "F4": "vocab / section guard failure",
    "F5": "sorter disagrees with BERT triage in verify mode",
    "F6": "reviewer disagrees with both / low confidence",
    "F7": "BERT runtime error (fail-soft to LLM)",
}


def _check_pass(handoff: dict[str, Any]) -> dict[str, tuple[bool, str]]:
    """P1-P5 from the handoff alone (self-contained checks)."""
    out: dict[str, tuple[bool, str]] = {}
    q = handoff.get("quality", {})
    triage = handoff.get("triage", {})

    out["P1"] = (
        handoff.get("method") == "bert"
        and bool(q.get("context_fit"))
        and float(q.get("coverage", 0.0)) == 1.0,
        f"method={handoff.get('method')} context_fit={q.get('context_fit')} "
        f"coverage={q.get('coverage')}",
    )
    out["P2"] = (
        not bool(q.get("messy")) and not q.get("guard_failures"),
        f"messy={q.get('messy')} guard_failures={q.get('guard_failures')}",
    )
    conf = triage.get("confidence", 0.0)
    gate = handoff.get("gate", {})
    thr = float(gate.get("bert_threshold", BERT_INTAKE_MIN_CONFIDENCE))
    out["P3"] = (float(conf) >= thr, f"confidence={conf} threshold={thr}")
    vocab_ok = bool(q.get("triage_vocab_ok"))
    out["P4"] = (vocab_ok, f"triage_vocab_ok={vocab_ok}")
    out["P5"] = (bool(q.get("section_map_ok")), f"section_map_ok={q.get('section_map_ok')}")
    return out


def _check_agreements(handoff: dict[str, Any], sorter_result: dict[str, Any] | None,
                      reviewer_result: dict[str, Any] | None, *,
                      mode: str) -> dict[str, tuple[bool, str]]:
    """P6/P7 — external-agreement checks (skip-mode + verify-mode aware).

    P6: in verify mode the sorter must agree with the BERT triage; in skip
    mode agreement is replaced by config authorization (allowlisted classes
    only, ``GATE_ALLOWLISTED_START`` — plan §12 rollout).  When the sorter
    did not run and mode is verify, P6 fails (verify requires the sorter).
    P7: when the reviewer fired, its doc_type must agree with the BERT
    triage or with the sorter (epic graph rule); no reviewer -> vacuously
    true (the reviewer is optional).
    """
    out: dict[str, tuple[bool, str]] = {}
    triage = handoff.get("triage", {})
    dt = triage.get("primary_doc_class")
    if mode == "skip":
        allow = GATE_ALLOWLISTED_START
        out["P6"] = (dt in allow, f"skip-mode allowlist={allow} class={dt}")
    elif sorter_result is None:
        out["P6"] = (False, "verify mode requires the sorter to have run")
    else:
        sorter_dt = sorter_result.get("doc_type")
        out["P6"] = (sorter_dt == dt, f"sorter={sorter_dt} bert={dt}")
    if reviewer_result is None:
        out["P7"] = (True, "reviewer did not fire (optional)")
    else:
        rev_dt = reviewer_result.get("doc_type")
        sorter_dt = sorter_result.get("doc_type") if sorter_result else None
        agree = rev_dt == dt or (sorter_dt is not None and rev_dt == sorter_dt)
        out["P7"] = (agree, f"reviewer={rev_dt} bert={dt} sorter={sorter_dt}")
    return out


def evaluate_intake_gate(handoff: dict[str, Any],
                         sorter_result: dict[str, Any] | None = None,
                         reviewer_result: dict[str, Any] | None = None, *,
                         mode: str | None = None) -> dict[str, Any]:
    """M5 gate: PASS only when ALL applicable P1-P7 hold; else FAIL with reasons.

    Failure recording (F1-F7, machine-readable): a FAIL carries the failed
    P-check plus any triggered F-check, mapped to the epic's recommended
    next action (force_llm_sorter / escalate_reviewer).  Reviewer blindness:
    the gate consumes the reviewer's verdict but never passes triage labels
    into the reviewer path — the reviewer result is an input, not an output.

    Never raises: a malformed handoff degrades to FAIL (F-code F7
    ``handoff_invalid``) — the caller keeps the run moving (D10).
    """
    mode = mode or _env_str("BERT_INTAKE_MODE", BERT_INTAKE_MODE)
    try:
        p_checks = {**_check_pass(handoff),
                    **_check_agreements(handoff, sorter_result,
                                        reviewer_result, mode=mode)}
    except Exception as exc:  # noqa: BLE001 — fail-open: gate never blocks
        return {
            "verdict": "FAIL",
            "mode": mode,
            "failures": ["F7"],
            "failed_checks": {"handoff_invalid": str(exc)},
            "recommended_action": "force_llm_sorter",
            "eligible_for_sorter_skip": False,
            "notes": "gate raised — degraded to LLM sorter (fail-open)",
        }

    failed = {k: detail for k, (ok, detail) in p_checks.items() if not ok}
    eligible = not failed
    # Epic next-action table: F5 (sorter disagrees) -> LLM sorter wins (reviewer
    # optional); F6 (reviewer disagrees) -> escalate the reviewer path.
    if "P7" in failed:
        recommended = "escalate_reviewer"
    elif "P6" in failed:
        recommended = "force_llm_sorter"
    elif mode == "skip" and eligible:
        recommended = "accept_bert"
    elif mode == "verify" and eligible:
        recommended = "verify_with_sorter"
    else:
        recommended = "force_llm_sorter"

    f_codes: list[str] = []
    if not p_checks["P1"][0]:
        f_codes.append("F1")
    if not p_checks["P2"][0]:
        f_codes.append("F2")
    if not p_checks["P3"][0]:
        f_codes.append("F3")
    if not p_checks["P4"][0] or not p_checks["P5"][0]:
        f_codes.append("F4")
    if not p_checks["P6"][0]:
        f_codes.append("F5")
    if not p_checks["P7"][0]:
        f_codes.append("F6")

    return {
        "verdict": "PASS" if eligible else "FAIL",
        "mode": mode,
        "checks": {k: {"ok": ok, "detail": detail}
                   for k, (ok, detail) in p_checks.items()},
        "failed_checks": failed,
        "failures": f_codes,
        "recommended_action": recommended,
        "eligible_for_sorter_skip": eligible,
        "notes": ("all PASS criteria hold" if eligible
                  else "PASS criteria unmet — LLM sorter remains authority"),
    }


# ---------------------------------------------------------------------------
# Env helpers (runtime overrides; committed defaults preserved)
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os_environ_get(name, default))
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    return os_environ_get(name, default)


def os_environ_get(name: str, default):
    import os

    return os.environ.get(name, default)
