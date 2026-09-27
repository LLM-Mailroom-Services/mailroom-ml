#!/usr/bin/env bash
# Idempotent: run complete_run.sh when a local M9a train has finished (no polling loop).
#
# Usage:
#   ./training/post_train_when_ready.sh              # check canonical + GPU1 smoke tags
#   ./training/post_train_when_ready.sh --run-tag TAG [--gpu N] [--min-epochs N]
#
# Typical overnight (cron every 42m or manual after TUI shows "trainer stopped"):
#   cd .../modernBERT && TORCH_DISABLE_NATIVE_JIT=1 ./training/post_train_when_ready.sh
#
# Respects logs/.m9a-gpu-policy: after GPU-1 smoke complete_run once, no further GPU-1 work.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export TORCH_DISABLE_NATIVE_JIT="${TORCH_DISABLE_NATIVE_JIT:-1}"

TAG_B="${M9A_PRIMARY_TAG:-m9a-local-20260927-014429}"
TAG_A="${M9A_GPU1_SMOKE_TAG:-m9a-local-gpu1-armA-1ep-20260927-025236}"
MARKER="logs/.m9a-gpu1-smoke-complete_run.done"

trainer_for_tag() {
  local tag="$1"
  pgrep -f "train_modernbert.py.*${tag}" 2>/dev/null | head -1 || true
}

epochs_run() {
  local tag="$1"
  local summary="data/modernbert_training/runs/${tag}/latest/summary.json"
  [[ -f "$summary" ]] || { echo 0; return; }
  "$ROOT/.venv/bin/python" -c "import json; print(json.load(open('$summary')).get('epochs_run', 0))"
}

try_complete() {
  local tag="$1" gpu="$2" min_ep="$3" skip_marker="${4:-}"
  local eval="reports/eval_${tag}.json"
  if [[ -n "$skip_marker" && -f "$MARKER" ]]; then
    echo "SKIP $tag (GPU1 smoke complete_run already done per $MARKER)"
    return 0
  fi
  if [[ -f "$eval" ]]; then
    echo "SKIP $tag (eval exists)"
    return 0
  fi
  if [[ -n "$(trainer_for_tag "$tag")" ]]; then
    echo "WAIT $tag (trainer still running)"
    return 0
  fi
  local er
  er="$(epochs_run "$tag")"
  if (( er < min_ep )); then
    echo "WAIT $tag (epochs_run=$er need>=$min_ep)"
    return 0
  fi
  echo "=== complete_run: $tag on GPU $gpu ==="
  local rc=0
  CUDA_VISIBLE_DEVICES="$gpu" ./training/complete_run.sh --run-tag "$tag" || rc=$?
  if [[ -f "$eval" ]]; then
    if [[ "$tag" == "$TAG_A" ]]; then
      touch "$MARKER"
    fi
    echo "NOTE $tag eval+gates done (complete_run exit=$rc; gate FAIL is ok for smoke compare)"
    return 0
  fi
  return "$rc"
}

CUSTOM_TAG=""
CUSTOM_GPU=""
CUSTOM_MIN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-tag=*) CUSTOM_TAG="${1#*=}" ;;
    --run-tag) CUSTOM_TAG="${2:-}"; shift ;;
    --gpu=*) CUSTOM_GPU="${1#*=}" ;;
    --gpu) CUSTOM_GPU="${2:-}"; shift ;;
    --min-epochs=*) CUSTOM_MIN="${1#*=}" ;;
    --min-epochs) CUSTOM_MIN="${2:-}"; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

if [[ -n "$CUSTOM_TAG" ]]; then
  gpu="${CUSTOM_GPU:-0}"
  min="${CUSTOM_MIN:-1}"
  try_complete "$CUSTOM_TAG" "$gpu" "$min"
  exit 0
fi

# Default batch: GPU1 smoke first (1 epoch), then primary Arm B (3 epochs).
try_complete "$TAG_A" 1 1 marker
try_complete "$TAG_B" 0 3
