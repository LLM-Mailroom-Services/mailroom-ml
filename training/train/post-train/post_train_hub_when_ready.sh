#!/usr/bin/env bash
# Eval + gates per run when train finishes; Hub publish only after BOTH runs are done.
#
# Usage (manual or cron / agent loop every 45m):
#   cd .../modernBERT && TORCH_DISABLE_NATIVE_JIT=1 ./training/train/post-train/post_train_hub_when_ready.sh
#
# Requires HF_TOKEN or HUGGING_FACE_HUB_TOKEN for real uploads (.env in repo root is sourced).
# Markers: logs/.m9a-hub-published-<run-tag>.done

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export TORCH_DISABLE_NATIVE_JIT="${TORCH_DISABLE_NATIVE_JIT:-1}"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

TAG_B="${M9A_PRIMARY_TAG:-m9a-local-20260927-014429}"
TAG_A="${M9A_GPU1_SMOKE_TAG:-m9a-local-gpu1-armA-1ep-20260927-025236}"
MIN_B="${M9A_PRIMARY_MIN_EPOCHS:-3}"
MIN_A="${M9A_GPU1_SMOKE_MIN_EPOCHS:-1}"

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

train_done() {
  local tag="$1" min_ep="$2"
  [[ -z "$(trainer_for_tag "$tag")" ]] || return 1
  local er
  er="$(epochs_run "$tag")"
  (( er >= min_ep ))
}

pub_marker() {
  echo "logs/.m9a-hub-published-${1}.done"
}

hf_hub_token_ready() {
  [[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]] && return 0
  "$ROOT/.venv/bin/python" -c "from huggingface_hub import get_token; raise SystemExit(0 if get_token() else 1)" \
    >/dev/null 2>&1
}

GOAL_STATUS="logs/.m9a-hub-goal-status.txt"
write_goal_status() {
  local hub_note="$1"
  {
    echo "updated: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "primary: $TAG_B trainer=$(trainer_for_tag "$TAG_B" || echo none) epochs_run=$(epochs_run "$TAG_B")"
    echo "smoke:   $TAG_A trainer=$(trainer_for_tag "$TAG_A" || echo none) epochs_run=$(epochs_run "$TAG_A")"
    echo "smoke_post_train: $([ -f logs/.m9a-gpu1-smoke-complete_run.done ] && echo done || echo pending)"
    echo "hub_auth: $(hf_hub_token_ready && echo ready || echo missing)"
    echo "hub_markers: $(ls logs/.m9a-hub-published-*.done 2>/dev/null | wc -l) of 2"
    echo "next_action: $hub_note"
  } >"$GOAL_STATUS"
}

echo "=== post_train_hub_when_ready $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "primary: $TAG_B (min epochs $MIN_B)"
echo "smoke:   $TAG_A (min epochs $MIN_A)"

# Eval + gates only (no Hub yet); respects GPU policy / smoke marker.
./training/eval/post_train_when_ready.sh

if ! train_done "$TAG_A" "$MIN_A"; then
  write_goal_status "wait smoke train/eval"
  echo "HUB WAIT: smoke $TAG_A still training or epochs incomplete"
  exit 0
fi
if ! train_done "$TAG_B" "$MIN_B"; then
  write_goal_status "wait primary train; then complete_run + hub publish both tags"
  echo "HUB WAIT: primary $TAG_B still training or epochs incomplete"
  exit 0
fi

for tag in "$TAG_A" "$TAG_B"; do
  marker="$(pub_marker "$tag")"
  eval_json="reports/eval_${tag}.json"
  if [[ -f "$marker" ]]; then
    echo "SKIP Hub: $tag already published ($marker)"
    continue
  fi
  if [[ ! -f "$eval_json" ]]; then
    echo "HUB WAIT: missing $eval_json for $tag (complete_run should have created it)"
    continue
  fi
  echo "=== Hub publish: $tag ==="
  pub_args=(--run-tag "$tag" --publish)
  # 1-epoch Arm A smoke is for λ_dt compare; #112 gates are expected to FAIL.
  if [[ "$tag" == "$TAG_A" ]]; then
    pub_args+=(--force-publish)
  fi
  rc=0
  ./training/train/complete_run.sh "${pub_args[@]}" || rc=$?
  if [[ -f "$marker" ]]; then
    continue
  fi
  if ! hf_hub_token_ready; then
    echo "HUB WAIT: no Hub token (export HF_TOKEN or: hf auth login)" >&2
    continue
  fi
  if (( rc == 0 )); then
    touch "$marker"
    echo "OK Hub marker -> $marker"
  else
    echo "FAIL Hub publish for $tag (exit $rc)" >&2
    exit "$rc"
  fi
done

write_goal_status "hub goal complete (both markers or skipped)"
echo "=== post_train_hub_when_ready: all requested publishes done or skipped ==="
echo "status file: $GOAL_STATUS"
