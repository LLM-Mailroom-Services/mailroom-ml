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
TRAINING_DATA_REVISION = "6790341e25229a2617914c839ff0e70c590d7b6d"

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
