#!/usr/bin/env bash
# One-shot exios66 Modal L4 eval over the full held-out-plus pool (1,323 docs).
# Deliberately minimal GPU surface: ONE deploy (bakes plus parquet) + ONE run.
#
# Prerequisites (env):
#   MODAL_TOKEN_ID, MODAL_TOKEN_SECRET (exios66 workspace) — required
#   HF_TOKEN — optional for heldout-plus (public corpus + pinned training data)
#
# Usage:
#   RUN_TAG=m9a-local-20260927-014429 MODULE=latest \
#     ./training/run_heldout_plus_modal_eval.sh
#
# Optional: SKIP_DEPLOY=1 if eval_app image already includes current plus build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${ROOT}/.venv/bin:${PATH}"

if [[ -f "${ROOT}/secrets.env" ]]; then
  # shellcheck disable=SC1091
  set -a && source "${ROOT}/secrets.env" && set +a
fi

if [[ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  _hf_cached="$(uv run python -c "from huggingface_hub import get_token; print(get_token() or '')" 2>/dev/null || true)"
  if [[ -n "${_hf_cached}" ]]; then
    export HF_TOKEN="${_hf_cached}"
    echo "[heldout-plus] HF_TOKEN from huggingface_hub cache (not logged)" >&2
  fi
  unset _hf_cached
fi
if [[ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "[heldout-plus] ERROR: HF_TOKEN required (private mailroom-finetune pin for eval)" >&2
  echo "  export HF_TOKEN=… or add to secrets.env (gitignored)" >&2
  exit 1
fi
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

export MODAL_PROFILE="${MODAL_PROFILE:-exios66}"
if [[ -n "${MODAL_TOKEN_ID:-}" && -n "${MODAL_TOKEN_SECRET:-}" ]]; then
  uv run --extra deploy modal token set \
    --token-id "${MODAL_TOKEN_ID}" \
    --token-secret "${MODAL_TOKEN_SECRET}" \
    --profile "${MODAL_PROFILE}" \
    --activate \
    --verify
else
  echo "[heldout-plus] MODAL_TOKEN_* unset — using existing Modal profile ${MODAL_PROFILE}" >&2
  if ! MODAL_PROFILE="${MODAL_PROFILE}" uv run --extra deploy modal app list >/dev/null 2>&1; then
    echo "ERROR: Modal profile ${MODAL_PROFILE} not authenticated — export MODAL_TOKEN_ID + MODAL_TOKEN_SECRET" >&2
    exit 1
  fi
fi

RUN_TAG="${RUN_TAG:-m9a-local-20260927-014429}"
MODULE="${MODULE:-latest}"
PLUS_DIR="${PLUS_DIR:-data/heldout_plus_v1}"
OUT_JSON="${OUT_JSON:-reports/eval_${RUN_TAG}-heldout-plus.json}"
BASELINE_JSON="${BASELINE_JSON:-reports/eval_${RUN_TAG}.json}"
SKIP_DEPLOY="${SKIP_DEPLOY:-0}"

ensure_plus_pool() {
  local audit="${PLUS_DIR}/audit.json"
  if [[ -f "${PLUS_DIR}/documents.parquet" && -f "$audit" ]]; then
    uv run python - <<'PY'
import json, sys
from pathlib import Path
audit = json.loads(Path("data/heldout_plus_v1/audit.json").read_text())
if not audit.get("gates_ok"):
    sys.exit("heldout_plus_v1 audit gates_ok is false — rebuild before GPU spend")
if audit.get("filter_stats", {}).get("selected") != 1000:
    sys.exit("expected 1000 plus docs in audit")
print("heldout_plus_v1 audit OK", flush=True)
PY
    return 0
  fi
  echo "[heldout-plus] building pool (CPU only)…" >&2
  uv run python training/dataset/mailroom-dataset/fetch_corpus.py
  uv run python training/dataset/enron/fetch_enron_heldout_inputs.py
  uv run python training/dataset/mailroom-dataset/build_heldout_plus.py \
    --enron-gt data/enrichment/enron_heldout/ground_truth/test.jsonl \
    --enron-blind data/enrichment/enron_heldout/blind/test.jsonl \
    --n 1000 --seed 42 --out "${PLUS_DIR}"
}

preflight_window_contract() {
  uv run python training/train/preflight_heldout_plus.py --plus-dir "${PLUS_DIR}"
}

ensure_plus_pool
preflight_window_contract

if [[ "$SKIP_DEPLOY" != "1" ]]; then
  echo "[heldout-plus] modal deploy eval_app (bakes ${PLUS_DIR})…" >&2
  uv run --extra deploy modal deploy deploy/eval_app.py
else
  echo "[heldout-plus] SKIP_DEPLOY=1 — assuming image already has plus mount" >&2
fi

mkdir -p reports reports/TEST-EVAL

echo "[heldout-plus] modal run: subset=heldout-plus sample=0 module=${MODULE}" >&2
uv run --extra deploy modal run deploy/eval_app.py \
  --module "${MODULE}" \
  --subset heldout-plus \
  --sample 0 \
  --seed 42 \
  --selective-risk \
  --as-json \
  --out "${OUT_JSON}"

uv run python - <<PY
import json, sys
from pathlib import Path
p = Path("${OUT_JSON}")
r = json.loads(p.read_text())
if r.get("eval_subset") != "heldout-plus":
    sys.exit(f"eval_subset mismatch: {r.get('eval_subset')}")
n = r.get("n_docs")
if n != 1323:
    sys.exit(f"expected n_docs=1323 got {n}")
print(f"GPU eval OK: n_docs={n} windows={r.get('window_calibration', {}).get('n_windows')}", flush=True)
PY

uv run python training/eval/interpret_heldout_plus_eval.py \
  --eval-json "${OUT_JSON}" \
  --baseline-json "${BASELINE_JSON}" \
  --run-tag "${RUN_TAG}" \
  --plus-dir "${PLUS_DIR}"

echo "[heldout-plus] done → ${OUT_JSON}" >&2
