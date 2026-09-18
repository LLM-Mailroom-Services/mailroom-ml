"""Routing-layer tests (mailroom-issues #85 M1/M2/M5, #86/#87/#90).

Hermetic: pure functions over dicts — no network, no model.  Truth tables
for the epic gate precedence (should_llm_intake), the M2 routing gate
(should_bert_intake) and the M5 P1-P7/F1-F7 PASS/FAIL wiring, plus the M1
handoff schema-v1 surface.
"""
from __future__ import annotations

from mailroom_ml.config import (
    BERT_INTAKE_MIN_CONFIDENCE,
    INTAKE_HANDOFF_SCHEMA_VERSION,
    MAX_TOKENS,
)
from mailroom_ml.routing import (
    build_handoff,
    context_fit,
    evaluate_intake_gate,
    should_bert_intake,
    should_llm_intake,
)

# ---------------------------------------------------------------------------
# should_llm_intake — epic gate precedence (plan §9)
# ---------------------------------------------------------------------------

def test_llm_gate_no_text_is_false():
    assert should_llm_intake("  ", {}, flag=1) is False


def test_llm_gate_messy_is_true():
    assert should_llm_intake("text", {"messy": True}, flag=1) is True


def test_llm_gate_flag_off_preserves_today():
    """Flag off: LLM only for messy/oversize — today's behavior (M6)."""
    assert should_llm_intake("short clean text", {}, flag=0) is False
    assert should_llm_intake("x" * 100_000, {}, flag=0) is True


def test_llm_gate_missing_ml_triage_fails_open():
    assert should_llm_intake("text", {}, ml_triage=None, flag=1) is True
    assert should_llm_intake("text", {}, ml_triage={"status": "failure"},
                             flag=1) is True


def test_llm_gate_failure_status_fails_open():
    triage = {"status": "failure", "route": "llm", "reason": "bert_error"}
    assert should_llm_intake("text", {}, ml_triage=triage, flag=1) is True


def test_llm_gate_non_fast_path_routes_llm():
    triage = {"status": "ok", "route": "llm", "reason": "gate_fail"}
    assert should_llm_intake("text", {}, ml_triage=triage, flag=1) is True


def test_llm_gate_fast_path_allows_skip():
    triage = {"status": "ok", "route": "fast_path", "reason": "fast_path"}
    assert should_llm_intake("text", {}, ml_triage=triage, flag=1) is False


def test_llm_gate_flag_off_ignores_ml_triage():
    """Flag off = today's behavior: short clean doc -> no LLM, even with a
    fast-path-capable triage present (BERT is not in charge when flag is 0)."""
    triage = {"status": "ok", "route": "fast_path", "reason": "fast_path"}
    assert should_llm_intake("short", {}, ml_triage=triage, flag=0) is False


# ---------------------------------------------------------------------------
# should_bert_intake — M2 routing gate (#87)
# ---------------------------------------------------------------------------

def test_bert_gate_flag_off_today():
    use, reason = should_bert_intake("text", {}, flag=0)
    assert (use, reason) == (False, "flag_off")


def test_bert_gate_empty_and_messy():
    assert should_bert_intake("", {}, flag=1) == (False, "empty")
    assert should_bert_intake("text", {"messy": True}, flag=1) == (False, "messy")


def test_bert_gate_oversize_chars_and_tokens():
    assert should_bert_intake("x" * 100_000, {}, flag=1) == (False, "oversize_chars")
    assert should_bert_intake("text", {"token_estimate": MAX_TOKENS + 1},
                              flag=1) == (False, "oversize_tokens")


def test_bert_gate_no_model_fails_open():
    assert should_bert_intake("text", {}, model_available=False, flag=1) == (
        False, "no_model")


def test_bert_gate_ok_within_context():
    assert should_bert_intake("a short notice.", {}, flag=1) == (
        True, "within_bert_ctx")


def test_bert_gate_mode_does_not_block_compute():
    """All modes compute BERT triage; the mode constrains the gate only."""
    for mode in ("shadow", "verify", "skip"):
        use, _ = should_bert_intake("text", {}, mode=mode, flag=1)
        assert use is True


def test_context_fit_budgets():
    assert context_fit("short")
    assert not context_fit("x" * 40_000)
    assert not context_fit("short", token_estimate=MAX_TOKENS + 500)


# ---------------------------------------------------------------------------
# M1 handoff schema v1 (#86)
# ---------------------------------------------------------------------------

def test_handoff_schema_v1_surface():
    h = build_handoff()
    assert h["schema_version"] == INTAKE_HANDOFF_SCHEMA_VERSION
    assert set(h) == {"schema_version", "method", "routing", "triage",
                      "sections", "quality", "gate", "provenance"}
    assert set(h["routing"]) == {"path", "reason", "bert_context_limit_tokens",
                                 "doc_chars", "doc_token_estimate"}
    assert set(h["triage"]) == {"primary_doc_class", "doc_subclass",
                                "confidence", "gist", "keywords"}
    assert set(h["quality"]) == {"messy", "context_fit", "coverage",
                                 "section_map_ok", "triage_vocab_ok",
                                 "guard_failures"}
    assert set(h["gate"]) == {"eligible_for_sorter_skip",
                              "recommended_action", "bert_threshold", "notes"}
    assert set(h["provenance"]) == {"model_id", "model_revision",
                                    "intake_windows", "changes_applied"}
    assert h["routing"]["bert_context_limit_tokens"] == MAX_TOKENS


def test_handoff_defaults_preserve_today():
    """Flag off -> clerk-only, not skip-eligible, force LLM sorter."""
    h = build_handoff()
    assert h["method"] == "deterministic"
    assert h["routing"]["path"] == "clerk_only"
    assert h["routing"]["reason"] == "flag_off"
    assert h["gate"]["eligible_for_sorter_skip"] is False
    assert h["gate"]["recommended_action"] == "force_llm_sorter"
    assert h["quality"]["triage_vocab_ok"] is False
    assert h["triage"]["primary_doc_class"] is None


def test_handoff_bert_surface():
    h = build_handoff(
        method="bert", routing_path="bert_intake", routing_reason="within_bert_ctx",
        doc_chars=1200, doc_token_estimate=300, doc_type="correspondence",
        doc_subclass="notice", confidence=0.97, gist="notice of action",
        keywords=["notice"], intake_windows=1, model_id="modernbert-base",
        eligible_for_sorter_skip=True, recommended_action="accept_bert",
        quality={"messy": False, "context_fit": True, "coverage": 1.0,
                 "triage_vocab_ok": True},
    )
    assert h["method"] == "bert"
    assert h["routing"]["path"] == "bert_intake"
    assert h["triage"]["primary_doc_class"] == "correspondence"
    assert h["triage"]["doc_subclass"] == "notice"
    assert h["quality"]["context_fit"] is True
    assert h["quality"]["coverage"] == 1.0
    assert h["gate"]["eligible_for_sorter_skip"] is True


# ---------------------------------------------------------------------------
# M5 evaluate_intake_gate — P1-P7 / F1-F7 (#90)
# ---------------------------------------------------------------------------

def _base_bert_handoff(**over) -> dict:
    h = build_handoff(
        method="bert", routing_path="bert_intake", routing_reason="within_bert_ctx",
        doc_chars=900, doc_token_estimate=220, doc_type="correspondence",
        doc_subclass="notice", confidence=0.97, intake_windows=1,
        model_id="modernbert-base",
        quality={"messy": False, "context_fit": True, "coverage": 1.0,
                 "section_map_ok": True, "triage_vocab_ok": True},
        eligible_for_sorter_skip=True, recommended_action="accept_bert",
    )
    for k, v in over.items():
        if k == "quality":
            h["quality"].update(v)
        elif k == "triage":
            h["triage"].update(v)
        elif k == "gate":
            h["gate"].update(v)
        elif k == "method":
            h["method"] = v
        else:
            h[k] = v
    return h


def test_gate_skip_mode_all_pass():
    h = _base_bert_handoff()
    gate = evaluate_intake_gate(h, mode="skip")
    assert gate["verdict"] == "PASS"
    assert gate["eligible_for_sorter_skip"] is True
    assert gate["recommended_action"] == "accept_bert"
    for p in ("P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        assert gate["checks"][p]["ok"] is True, gate["checks"][p]


def test_gate_skip_mode_allowlist_governs_p6():
    h = _base_bert_handoff(triage={"primary_doc_class": "contract"})
    gate = evaluate_intake_gate(h, mode="skip")
    assert gate["verdict"] == "FAIL"
    assert gate["failures"] == ["F5"]
    assert "P6" in gate["failed_checks"]
    assert gate["eligible_for_sorter_skip"] is False


def test_gate_p1_requires_method_context_coverage():
    h = _base_bert_handoff(method="llm")
    assert evaluate_intake_gate(h, mode="skip")["verdict"] == "FAIL"
    h = _base_bert_handoff(quality={"context_fit": False})
    assert evaluate_intake_gate(h, mode="skip")["verdict"] == "FAIL"
    h = _base_bert_handoff(quality={"coverage": 0.5})
    assert evaluate_intake_gate(h, mode="skip")["verdict"] == "FAIL"


def test_gate_p2_messy_or_guard_failures():
    h = _base_bert_handoff(quality={"messy": True})
    gate = evaluate_intake_gate(h, mode="skip")
    assert gate["verdict"] == "FAIL" and "P2" in gate["failed_checks"]
    h = _base_bert_handoff(quality={"guard_failures": ["vocab_fail"]})
    gate = evaluate_intake_gate(h, mode="skip")
    assert gate["verdict"] == "FAIL" and "P2" in gate["failed_checks"]


def test_gate_p3_confidence_threshold():
    h = _base_bert_handoff(triage={"confidence": BERT_INTAKE_MIN_CONFIDENCE - 0.1})
    gate = evaluate_intake_gate(h, mode="skip")
    assert gate["verdict"] == "FAIL" and "P3" in gate["failed_checks"]


def test_gate_p4_p5_guard_checks():
    h = _base_bert_handoff(quality={"triage_vocab_ok": False})
    assert "P4" in evaluate_intake_gate(h, mode="skip")["failed_checks"]
    h = _base_bert_handoff(quality={"section_map_ok": False})
    assert "P5" in evaluate_intake_gate(h, mode="skip")["failed_checks"]


def test_gate_verify_mode_requires_sorter_runs_and_agrees():
    # verify mode without a sorter result -> P6 fail
    h = _base_bert_handoff()
    gate = evaluate_intake_gate(h, mode="verify")
    assert gate["verdict"] == "FAIL"
    assert "P6" in gate["failed_checks"]
    # sorter agrees -> PASS
    gate = evaluate_intake_gate(
        h, sorter_result={"doc_type": "correspondence"}, mode="verify")
    assert gate["verdict"] == "PASS"
    assert gate["recommended_action"] == "verify_with_sorter"
    # sorter disagrees -> F5, LLM authority
    gate = evaluate_intake_gate(
        h, sorter_result={"doc_type": "contract"}, mode="verify")
    assert gate["verdict"] == "FAIL"
    assert "F5" in gate["failures"]
    assert gate["recommended_action"] == "force_llm_sorter"


def test_gate_reviewer_blind_and_agreement():
    """P7: reviewer doc_type agrees with BERT or sorter (blind — the reviewer
    result is an input here, never a surface the gate hands back)."""
    h = _base_bert_handoff()
    gate = evaluate_intake_gate(h, sorter_result={"doc_type": "correspondence"},
                                reviewer_result={"doc_type": "correspondence"},
                                mode="verify")
    assert gate["verdict"] == "PASS" and gate["checks"]["P7"]["ok"] is True
    gate = evaluate_intake_gate(h, sorter_result={"doc_type": "correspondence"},
                                reviewer_result={"doc_type": "insurance_claim"},
                                mode="verify")
    assert gate["verdict"] == "FAIL"
    assert "F6" in gate["failures"]
    assert gate["recommended_action"] == "escalate_reviewer"


def test_gate_never_raises_on_garbage():
    """Fail-open: malformed handoffs degenerate to FAIL, never raise."""
    for junk in (None, {}, {"quality": None}, {"triage": "nope"},
                 {"gate": {"bert_threshold": "x"}}):
        gate = evaluate_intake_gate(junk, mode="verify")
        assert gate["verdict"] == "FAIL"
        assert gate["eligible_for_sorter_skip"] is False


def test_gate_skip_eligibility_only_when_all_pass():
    """Skip eligibility requires EVERY P — one failure kills the skip."""
    h = _base_bert_handoff(triage={"confidence": 0.5})
    gate = evaluate_intake_gate(h, sorter_result={"doc_type": "correspondence"},
                                mode="verify")
    assert gate["verdict"] == "FAIL"
    assert gate["eligible_for_sorter_skip"] is False
    assert gate["recommended_action"] == "force_llm_sorter"
