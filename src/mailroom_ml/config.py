"""mailroom-ml central configuration — the interlock every module imports.

All Hub pins are verified against the live Hub API (2026-09-18):

- ``mailroom-dataset`` (canonical, immutable eval corpus) @
  46a4d3c240a36671cde0182fff4960f6b8b73aca — the evals harness pin.
- ``mailroom-finetune`` (the working duplicate for ML) @
  19720ceb4e29bc3134a88507aa57cdfac7a64a1b — created 2026-09-18, byte-schema
  identical to the canonical corpus (36-col ``ground_truth`` incl.
  matter/group/relationships + gt_fields, same 3,302 rows /
  2,979 train / 323 test).  This is the repo the user calls the
  "mailroom-train" duplicate.
- ``mailroom-modernbert-training`` (prepared training set from the committed
  corpus-eda work, cf096fa) @ 6790341e25229a2617914c839ff0e70c590d7b6d.

Model + windowing constants mirror the committed ``modernbert/prep.py``
(Mailroom-Corpus-EDA, commit cf096fa) so the published training set stays
byte-compatible with this build.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Repo / paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
STAGE_DIR = DATA_DIR / "modernbert_training" / "stage"
RUNS_DIR = DATA_DIR / "modernbert_training" / "runs"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

# ---------------------------------------------------------------------------
# Hub pins (verified 2026-09-18 against api.huggingface.co)
# ---------------------------------------------------------------------------
CANONICAL_REPO = "Lucius-Morningstar/mailroom-dataset"
CANONICAL_REVISION = "46a4d3c240a36671cde0182fff4960f6b8b73aca"

# The modifiable working copy — change this one constant if the duplicate is
# ever re-homed under a different repo id.
FINETUNE_REPO = "Lucius-Morningstar/mailroom-finetune"
FINETUNE_REVISION = "19720ceb4e29bc3134a88507aa57cdfac7a64a1b"

TRAINING_DATA_REPO = "Lucius-Morningstar/mailroom-modernbert-training"
# Republished 2026-09-20 (pre-flight clean): the 2026-09-19 build leaked the
# label through the title fallback (title==filename for 65.9% of rows; 42.8%
# of filenames carried the subclass token) and consumed RAW corpus text while
# the pipeline feeds the classifier deterministic_normalize(doc_text).  This
# revision is leak-free (semantic-only titles) and clerk-normalized, so
# training input is byte-representative of inference input.
TRAINING_DATA_REVISION = "5b72a345cd3c057b736bea4910fdbef6509ad1c3"

# Operator-set at publish time (never a default in committed code paths).
CLASSIFIER_MODEL_REPO = "Lucius-Morningstar/mailroom-modernbert-classifier"

# ---------------------------------------------------------------------------
# Model + windowing (committed corpus-eda constants)
# ---------------------------------------------------------------------------
MODEL_ID = "answerdotai/ModernBERT-base"
MAX_TOKENS = 8192                 # ModernBERT native context
WINDOW_OVERLAP_TOKENS = 512
VAL_FRACTION = 0.1
RANDOM_STATE = 42                 # every split / shuffle / generator seed base

# Extra seeds for the final-candidate discipline (baseline + 2 more).
TRAINING_SEEDS = (42, 7, 2026)

# ---------------------------------------------------------------------------
# Taxonomies (canonical keys vendored from llm-dojo-scoring, DMR-066 — the
# same normalization the sandbox corpus uses; see mailroom_ml.labels)
# ---------------------------------------------------------------------------
DOC_TYPES: tuple[str, ...] = (
    "contract", "merger_agreement", "corporate_record", "correspondence",
    "insurance_claim",
)

# Canonical subclass surfaces sanctioned by the constellation tracker
# (#66 critical: correspondence is 8-key — zero voicemail, zero other; #67:
# corporate_record observed GT is 10 keys — certificate_of_formation has
# zero rows; #68: insurance is 6-key). Head label maps are DERIVED from the
# observed GT at build time (surface-drift parity test, #66/#67/#75); these
# tuples are the canonical normalization references, not the head vocab.
CANONICAL_CORRESPONDENCE_SUBCLASSES: tuple[str, ...] = (
    "email", "memo", "letter", "notice", "demand", "attorney_demand",
    "press_release", "meeting_request",
)
OBSERVED_CORPORATE_RECORD_SUBCLASSES: tuple[str, ...] = (
    "charter_amendment", "articles_of_incorporation", "officer_certificate",
    "indenture", "subsidiary_list", "rights_instrument", "board_resolution",
    "bylaws", "powers_of_attorney", "other",
)
INSURANCE_SUBCLASSES: tuple[str, ...] = (
    "carrier", "inpatient", "outpatient", "pde", "property", "auto",
)

# #107 taxonomy-conformance projections: canonical subclass surfaces the
# trained heads do NOT carry must project deterministically at inference
# (never force-fit into a sibling class, never crash on an unknown key).
#   - corporate_record: canonical 11-key surface vs the observed 10-key head
#     (#67 — certificate_of_formation has zero observed rows) -> project to
#     the head's `other` when the head carries one.
#   - insurance_claim: 6-key head with NO `other` class -> any canonical
#     subclass outside the head vocab is UNMAPPED and routes to the LLM
#     (SUBCLASS_UNMAPPED_ROUTE), never silently re-mapped.
SUBCLASS_PROJECTIONS: dict[str, dict[str, str]] = {
    "corporate_record": {"certificate_of_formation": "other"},
}
SUBCLASS_UNMAPPED_ROUTE = "llm"

# ---------------------------------------------------------------------------
# Routing policy — PLACEHOLDERS until selective-risk analysis on the
# calibration set quantifies the right thresholds (Plan §Calibration).  The
# pipeline's existing 0.88/0.97 confidence bands are preserved as the initial
# deployment policy; the composite route score S = p_calibrated * agreement *
# margin must clear the fast-path gate below.
# ---------------------------------------------------------------------------
ROUTE_DOC_CONFIDENCE = 0.97        # calibrated doc_type confidence (initial)
ROUTE_SUBCLASS_CONFIDENCE = 0.95   # calibrated subclass confidence (initial)
ROUTE_WINDOW_AGREEMENT = 0.80      # supporting-window share
ROUTE_MARGIN = 0.10                # winner minus runner-up, normalized
ROUTE_MIN_AUTHENTIC_SUPPORT = 5    # real training rows required per label
FAST_PATH_ERROR_BUDGET = 0.02      # selective-risk target: P(err | fast path)

# Abstention / unknown + OOD gate (novelty flag from OOD probe, Phase 2).
ABSTAIN_UNKNOWN_CLASS = "unknown"

# ---------------------------------------------------------------------------
# Constellation intake-overhaul contract (mailroom-issues #85 epic, M1-M7 —
# the org-owned spec this repo's routing layer must satisfy; the governed
# llm-mailroom graph consumes these, mailroom-ml only defines the contract).
# ---------------------------------------------------------------------------
# Feature flag + mode mirror the org contract verbatim:
#   MAILROOM_BERT_INTAKE=0   -> today's behavior (instant rollback)
#   BERT_INTAKE_MODE=shadow  -> compute always, sorter always runs (metrics)
#   BERT_INTAKE_MODE=verify  -> sorter runs, PASS requires agreement (P6)
#   BERT_INTAKE_MODE=skip    -> PASS skips the LLM sorter (allowlisted classes)
MAILROOM_BERT_INTAKE_DEFAULT = 0          # default OFF preserves pipeline behavior
BERT_INTAKE_MODE = "shadow"               # rollout ladder: shadow -> verify -> skip

# Initial routing thresholds from #85/#89; the #84-calibrated value replaces
# BERT_INTAKE_MIN_CONFIDENCE once selective-risk analysis lands (plan S8).
BERT_INTAKE_MIN_CONFIDENCE = 0.92         # calibrated doc_type confidence (epic example value)
BERT_INTAKE_MAX_CHARS = 30_000            # ~8,192 tokens at ~3.8 chars/token (context_fit gate)
INTAKE_HANDOFF_SCHEMA_VERSION = 1         # intake_handoff schema v1 (M1)

# PASS/FAIL gate vocabulary (epic P1-P7 / F1-F7) — used by routing/evaluate_gate.
GATE_REQUIRED_AGREEMENT = 0.80            # window-agreement floor (plan S = p*a*m)
GATE_ALLOWLISTED_START = ("correspondence",)  # M7: first classes eligible for skip

# ---------------------------------------------------------------------------
# Enrichment source pins (Hub API, verified 2026-09-18) — per #52 discipline,
# every source pool is revision-pinned; never pull live tips.
# ---------------------------------------------------------------------------
ENRON_DEDUP_REPO = "Lucius-Morningstar/enron-correspondence-dedup"
ENRON_DEDUP_REVISION = "993919b4387f017b2fcff5902102de609ad41464"   # 247,523 rows
CMS_POOL_REPO = "Lucius-Morningstar/cms-desynpuf-insurance-claims"
CMS_POOL_REVISION = "875da3aa1c4cf0220be7e3fe11e489ec7490e840"      # 400 rendered
GNOTHEIA_REPO = "gratex/GNOTHEIA-synthetic-insurance-dataset"
GNOTHEIA_REVISION = "c006552404f8dc5bea89de5b39ecf4672607acef"      # 863 polycontexts
BDR_REPO = "bdr-ai-org/insurance-motor-claims-decision-v1"
BDR_REVISION = "090163351d02a0f7d5d4b5143aec6bcf878e2d59"           # 800 tabular
INSURBIAS_REPO = "feihuangfh/INSURBIAS"
INSURBIAS_REVISION = "311d59c4d2e117db3f4f12f515c0536090b4d876"     # narratives CSV
CUAD_FULL_REPO = "Lucius-Morningstar/mailroom-cuad-contracts-full"
CUAD_FULL_REVISION = "e69afe340b48133b7173d74b8ad220fcd28a1a6e"

# ---------------------------------------------------------------------------
# Synthetic-data program caps (policy v1 — see configs/synthetic_policy_v1.yaml)
# ---------------------------------------------------------------------------
SYNTHETIC_MAX_GLOBAL_SHARE = 0.40
SYNTHETIC_MAX_PER_SUBCLASS_SHARE = 0.50
SYNTHETIC_EXAMPLE_WEIGHT = 0.6     # loss weight for synthetic rows (0.4-0.7)
SYNTHETIC_ELIGIBILITY_MAX_AUTHENTIC = 20
SYNTHETIC_TIER_CAPS = ((5, 3), (15, 2), (30, 1), (75, 0))  # (authentic, cap)

# Input-construction version: v1 = committed title + "\\n\\n" + window
# (byte-compatible with the published training set).  v2 adds the tagged
# [FILE_NAME]/[TITLE]/[WINDOW_INDEX] metadata prefix — config-flag only.
INPUT_FORMAT_VERSION = "v1"
