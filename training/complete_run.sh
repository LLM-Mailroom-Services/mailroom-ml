#!/usr/bin/env bash
# Post-train pipeline for a finished M9a local run (read-only w.r.t. training).
#
# Never starts, stops, or signals train_modernbert. If the checkpoint is not
# ready (no summary.json), prints status and exits without touching a GPU.
#
# Usage:
#   ./training/complete_run.sh --run-tag m9a-local-20260927-014429
#   ./training/complete_run.sh --run-tag m9a-local-20260927-014429 --publish
#   ./training/complete_run.sh --run-tag m9a-local-20260927-014429 --publish --dry-run
#   ./training/complete_run.sh --run-tag m9a-local-20260927-014429 --publish --force-publish
#
# Steps (idempotent):
#   1. Verify data/modernbert_training/runs/<tag>/latest/summary.json exists
#   2. Run eval_modernbert.py if reports/eval_<tag>.json is missing
#      (subclass/doc_type calibration surfaces + routing thresholds)
#   3. Run check_m9a_gates.py (#112 held-out test gates)
#   4. With --publish: call publish_run_to_hub.sh when HF_TOKEN is set and
#      gates PASS (or --force-publish / publish --ignore-gates)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || { echo "missing Linux venv at .venv" >&2; exit 1; }

RUN_TAG=""
PUBLISH=0
DRY_RUN=0
FORCE_PUBLISH=0
MIN_FREE_MIB=512

usage() {
  sed -n '2,19p' "$0"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-tag=*) RUN_TAG="${1#*=}" ;;
    --run-tag) RUN_TAG="${2:-}"; shift ;;
    --publish) PUBLISH=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --force-publish) FORCE_PUBLISH=1 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
  shift
done

if [[ -z "$RUN_TAG" ]]; then
  echo "ERROR: --run-tag is required" >&2
  usage 1
fi

CKPT="data/modernbert_training/runs/${RUN_TAG}/latest"
SUMMARY="${CKPT}/summary.json"
EVAL_JSON="reports/eval_${RUN_TAG}.json"

echo "=== complete_run: ${RUN_TAG} ==="
echo "checkpoint -> ${CKPT}"
echo "summary    -> ${SUMMARY}"
echo "eval       -> ${EVAL_JSON}"

if [[ ! -f "$SUMMARY" ]]; then
  echo "WAIT: ${SUMMARY} missing — training not finished or wrong --run-tag." >&2
  echo "Monitor only: ./training/watch_m9a_run.sh --log logs/${RUN_TAG}.log --agent" >&2
  exit 4
fi

trainer_on_ckpt=0
for pid in $(pgrep -f 'train_modernbert\.py' 2>/dev/null || true); do
  args="$(ps -o args= -p "$pid" 2>/dev/null || true)"
  if [[ "$args" == *"${CKPT}"* || "$args" == *"runs/${RUN_TAG}/"* ]]; then
    trainer_on_ckpt=1
    echo "WAIT: train_modernbert still running for this run (PID ${pid})." >&2
    echo "Re-run complete_run after training exits; this script will not stop it." >&2
    echo "Monitor (42m polls): ./training/watch_m9a_run.sh --log logs/${RUN_TAG}.log --agent" >&2
    exit 5
  fi
done
pick_eval_gpu() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *","* ]]; then
    local idx="$CUDA_VISIBLE_DEVICES"
    local used
    used=$(nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo 999999)
    if (( used < MIN_FREE_MIB )); then
      echo "$idx"
      return 0
    fi
    echo "CUDA_VISIBLE_DEVICES=$idx in use (${used} MiB)" >&2
    return 1
  fi
  local line idx used
  while IFS=, read -r idx used; do
    idx=$(echo "$idx" | tr -d ' ')
    used=$(echo "$used" | tr -d ' ')
    if (( used < MIN_FREE_MIB )); then
      echo "$idx"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
  return 1
}

if [[ ! -f "$EVAL_JSON" ]]; then
  if ! GPU_IDX="$(pick_eval_gpu)"; then
    echo "ERROR: no free GPU for eval (need <${MIN_FREE_MIB} MiB used). Set CUDA_VISIBLE_DEVICES or wait." >&2
    exit 6
  fi
  echo "=== held-out eval + calibration (GPU ${GPU_IDX}) ==="
  mkdir -p reports
  CUDA_VISIBLE_DEVICES="$GPU_IDX" "$PY" training/eval_modernbert.py \
    --checkpoint "$CKPT" \
    --stage data/modernbert_training/stage \
    --sample 0 \
    --selective-risk \
    --write-routing-thresholds "${CKPT}/routing_thresholds.json" \
    --json >"$EVAL_JSON"
  echo "eval report -> ${EVAL_JSON}"
else
  echo "SKIP eval: ${EVAL_JSON} already exists"
fi

echo "=== M9a #112 gates + subclass/doc_type calibration report ==="
GATE_RC=0
"$PY" training/check_m9a_gates.py "$EVAL_JSON" "$SUMMARY" || GATE_RC=$?

if (( PUBLISH == 0 )); then
  echo "=== publish ==="
  echo "  (pass --publish to upload after gates PASS; --dry-run for plan only)"
  exit "$GATE_RC"
fi

PUBLISH_ARGS=(
  ./training/publish_run_to_hub.sh
  --checkpoint "$CKPT"
  --eval-json "$EVAL_JSON"
  --release-tag "$RUN_TAG"
)
(( DRY_RUN == 1 )) && PUBLISH_ARGS+=(--dry-run)
(( FORCE_PUBLISH == 1 )) && PUBLISH_ARGS+=(--ignore-gates)
# Gates already checked above; allow dry-run publish plan even when FAIL.
if (( DRY_RUN == 1 && GATE_RC != 0 )); then
  PUBLISH_ARGS+=(--ignore-gates)
fi

if (( GATE_RC != 0 )) && (( FORCE_PUBLISH == 0 )) && (( DRY_RUN == 0 )); then
  echo "SKIP Hub publish: M9a gates not met (use --force-publish or fix eval)" >&2
  exit "$GATE_RC"
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "SKIP Hub publish: HF_TOKEN not set — dry-run plan:" >&2
  "${PUBLISH_ARGS[@]}" --dry-run
  exit "$GATE_RC"
fi

if (( DRY_RUN == 1 )); then
  "${PUBLISH_ARGS[@]}"
  exit "$GATE_RC"
fi

"${PUBLISH_ARGS[@]}"
exit "$GATE_RC"
