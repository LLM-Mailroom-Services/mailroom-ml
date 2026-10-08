#!/usr/bin/env bash
# Local M9a subclass-improvement training on THIS server (not Modal).
#
# Gated: refuses to touch a GPU unless you pass --i-authorize-gpu.
# Uses only free GPUs (default: first idle GPU with <512 MiB used).
# Never kills or displaces existing jobs (e.g. vLLM on another GPU).
#
# Usage:
#   ./training/run_m9a_local.sh                  # dry-run / status only
#   ./training/run_m9a_local.sh --i-authorize-gpu # foreground: train (tee) + post-run eval
#   ./training/run_m9a_local.sh --i-authorize-gpu --background  # train only via nohup; watch separately
#   ./training/logging/watch_m9a_run.sh --follow --pretty  # kawaii TTY monitor (plain if piped / NO_COLOR)
#   ./training/train/complete_run.sh --run-tag <RUN_TAG> [--publish]  # after train: eval + #112 gates + Hub
#   ./training/run_m9a_local.sh --i-authorize-gpu --publish-to-hub  # + Hub release if gates pass
#   CUDA_VISIBLE_DEVICES=0 ./training/run_m9a_local.sh --i-authorize-gpu
#   ./training/run_m9a_local.sh --i-authorize-gpu --arm=m9b  # head-LR + logit-adjust arm (#43)
#
# Canonical paths:
#   - Interactive full pipeline: foreground --i-authorize-gpu (blocks until train+eval done).
#   - Agents / long runs: --background then watch_m9a_run.sh (do not spawn a second launcher).
#   Duplicate launch is refused if logs/.m9a-train.lock is held or train_modernbert is already running.
#
# Arm B (M9a handoff) + post-run-3 fixes on main:
#   inverse weights / cap 20, subclass CE label smoothing, select-on-subclass,
#   mlp heads, per-head temperature calibration in the trainer, eval harness
#   for subclass macro-F1 + routing thresholds.
#
# Artifacts (after an authorized run):
#   logs/<run_tag>.log                 — full stdout/stderr tee (incl. ETA lines)
#   <run_dir>/latest/                  — live checkpoint + summary.json
#   <run_dir>/latest/resume.json       — optimizer counters for --resume
#   <run_dir>/latest/resume_manifest.json — latest resumable path + CLI hint
#   <run_dir>/latest/train_steps.jsonl — step loss / steps/s / ETA (monitoring)
#   <run_dir>/latest/epoch_metrics.{jsonl,tsv} — epoch table + selection
#   <run_dir>/runs/<run_id>-eN/        — per-epoch checkpoint archives
#   reports/eval_<run_tag>.json        — held-out eval (post-train)
#
# Resume (after kill or crash — reuse the SAME hyperparameters as the run):
#   CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python training/train/train_modernbert.py \
#     --data data/modernbert_training/stage \
#     --output data/modernbert_training/runs/<run_tag>/latest \
#     --resume data/modernbert_training/runs/<run_tag>/latest \
#     ...same flags as below...
# Mid-epoch checkpoints land every --log-every micro-batches (25 here) once
# the trainer process is on code that writes them; epoch-end always resumable.
#
# ETA: each trainer step line prints "ETA …"; also in train_steps.jsonl
# (`eta` / `eta_s`) and epoch_metrics (`eta_remaining`).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || { echo "missing Linux venv at .venv — recreate with python3.11 -m venv .venv && pip install -e '.[train,dev]'" >&2; exit 1; }

AUTHORIZE=0
PUBLISH=0
BACKGROUND=0
ARM="b"
RESUME_OUT=""
EPOCHS=4
LOG_EVERY=25
# Effective batch = BATCH × GRAD_ACCUM (samples per optimizer step).
# Modal L4 (~22 GB): 4 × 8 = 32 — tight VRAM at max_length 8192 + sdpa + checkpointing.
# Local RTX A5000 24 GB: 8 × 4 = 32 — same effective batch as Modal; ~7.9 GB VRAM
# with grad checkpointing @ 8192. (--no-gradient-checkpointing OOMs by batch 8.)
BATCH=8
GRAD_ACCUM=4
# Faster kernels; not bitwise-identical to seed=42 deterministic cudnn (Modal keeps off).
CUDNN_BENCHMARK=1
CHECKPOINT_EVERY=200
MIN_FREE_MIB=512
LOCK_FILE="${ROOT}/logs/.m9a-train.lock"

trainer_pids() {
  local pid args
  for pid in $(pgrep -f 'train_modernbert\.py' 2>/dev/null || true); do
    args="$(ps -o args= -p "$pid" 2>/dev/null || true)"
    if [[ "$args" == *python*train_modernbert.py* ]]; then
      echo "$pid"
    fi
  done
}

refuse_if_training_active() {
  local existing
  existing="$(trainer_pids)"
  if [[ -n "$existing" ]]; then
    echo "Refusing launch: train_modernbert already running (PIDs: $(echo "$existing" | tr '\n' ' '))." >&2
    echo "Use ./training/logging/watch_m9a_run.sh --follow — monitors only; does not start jobs." >&2
    exit 3
  fi
  if [[ -f "$LOCK_FILE" ]]; then
    local lock_pid rest
    read -r lock_pid rest < "$LOCK_FILE" || true
    if [[ -n "${lock_pid:-}" ]] && kill -0 "$lock_pid" 2>/dev/null; then
      echo "Refusing launch: lock $LOCK_FILE held by PID $lock_pid ($rest)." >&2
      exit 3
    fi
    rm -f "$LOCK_FILE"
  fi
}

write_lock() {
  local pid="$1"
  mkdir -p logs
  echo "$pid $RUN_TAG $LOG" >"$LOCK_FILE"
}

clear_lock_if_ours() {
  if [[ -f "$LOCK_FILE" ]]; then
    local lock_pid
    read -r lock_pid _ < "$LOCK_FILE" || true
    if [[ "$lock_pid" == "$1" ]]; then
      rm -f "$LOCK_FILE"
    fi
  fi
}

for arg in "$@"; do
  case "$arg" in
    --i-authorize-gpu) AUTHORIZE=1 ;;
    --publish-to-hub) PUBLISH=1 ;;
    --background) BACKGROUND=1 ;;
    --epochs=*) EPOCHS="${arg#*=}" ;;
    --batch-size=*) BATCH="${arg#*=}" ;;
    --grad-accum=*) GRAD_ACCUM="${arg#*=}" ;;
    --log-every=*) LOG_EVERY="${arg#*=}" ;;
    --resume=*) RESUME_OUT="${arg#*=}" ;;
    --cudnn-benchmark) CUDNN_BENCHMARK=1 ;;
    --arm=*) ARM="${arg#*=}" ;;
    --help|-h)
      sed -n '2,40p' "$0"
      exit 0
      ;;
  esac
done

echo "=== GPU status (read-only) ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv || true

pick_free_gpu() {
  # Prefer CUDA_VISIBLE_DEVICES if already set to a single idle index.
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *","* ]]; then
    local idx="$CUDA_VISIBLE_DEVICES"
    local used
    used=$(nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if (( used < MIN_FREE_MIB )); then
      echo "$idx"
      return 0
    fi
    echo "CUDA_VISIBLE_DEVICES=$idx is in use (${used} MiB)" >&2
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
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  return 1
}

if ! GPU_IDX="$(pick_free_gpu)"; then
  echo "No free GPU window (need <${MIN_FREE_MIB} MiB used). Not starting." >&2
  exit 2
fi
echo "Selected free GPU index: $GPU_IDX"

if [[ -n "$RESUME_OUT" ]]; then
  _resume_path="$RESUME_OUT"
  [[ "$_resume_path" != /* ]] && _resume_path="$ROOT/$_resume_path"
  _resume_path="$(cd "$_resume_path" && pwd)"
  OUT="$(realpath --relative-to="$ROOT" "$_resume_path")"
  RUN_DIR="$(dirname "$OUT")"
  RUN_TAG="$(basename "$RUN_DIR")"
  LOG="logs/${RUN_TAG}.log"
  mkdir -p logs "$OUT" "$RUN_DIR/runs" reports
else
  RUN_TAG="m9a-local-$(date +%Y%m%d-%H%M%S)"
  RUN_DIR="data/modernbert_training/runs/${RUN_TAG}"
  OUT="${RUN_DIR}/latest"
  LOG="logs/${RUN_TAG}.log"
  mkdir -p logs "$OUT" "$RUN_DIR/runs" reports
fi

CMD=(
  "$PY" training/train/train_modernbert.py
  --data data/modernbert_training/stage
  --output "$OUT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH"
  --grad-accum "$GRAD_ACCUM"
  --lr 2e-5
  --seed 42
  --eval-test
  --mlp-heads
  --label-smoothing 0.05
  --subclass-label-smoothing 0.05
  --max-length 8192
  --weight-decay 0.01
  --warmup-frac 0.06
  --early-stop-patience 2
  --loss-lambda-dt 0.65
  --log-every "$LOG_EVERY"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --prefetch-batches 2
  # --select-on-subclass is default on (lexicographic subclass objective)
)
case "$ARM" in
  # Arm B (M9a pre-registered): inverse class weights, cap 20.
  b) CMD+=(--weight-mode inverse --weight-cap 20) ;;
  # M9b (#43): subclass heads at their own LR + logit-adjusted CE, no
  # per-row weights. The heads shared the encoder's 2e-5 LR and collapsed
  # to a constant answer per head in Arm A/B.
  m9b) CMD+=(--weight-mode none --subclass-head-lr 1e-3
             --subclass-logit-adjust 1.0) ;;
  *) echo "unknown --arm=$ARM (expected b or m9b)" >&2; exit 2 ;;
esac
if (( CUDNN_BENCHMARK == 1 )); then
  CMD+=(--cudnn-benchmark)
fi
if [[ -n "$RESUME_OUT" ]]; then
  CMD+=(--resume "$OUT")
fi

echo "=== Planned command (GPU $GPU_IDX) ==="
printf 'CUDA_VISIBLE_DEVICES=%s' "$GPU_IDX"
for _a in "${CMD[@]}"; do printf ' %q' "$_a"; done
printf '\n'
echo "log            -> $LOG"
echo "out (latest)   -> $OUT"
echo "epoch archives -> ${RUN_DIR}/runs/<run_id>-eN"
echo "step jsonl     -> $OUT/train_steps.jsonl"
echo "epoch metrics  -> $OUT/epoch_metrics.jsonl (+ .tsv)"
echo "summary        -> $OUT/summary.json"
echo "ETA            -> step lines in $LOG and train_steps.jsonl"
echo "resume         -> add --resume $OUT to the trainer (see script header)"
RESUME_CMD=( "${CMD[@]}" --resume "$OUT" )
echo "=== Resume command (after interrupt; same GPU/hyperparams) ==="
printf 'CUDA_VISIBLE_DEVICES=%s' "$GPU_IDX"
for _a in "${RESUME_CMD[@]}"; do printf ' %q' "$_a"; done
printf '\n'

if (( AUTHORIZE != 1 )); then
  echo
  echo "DRY RUN ONLY — GPU not claimed."
  echo "Re-run with --i-authorize-gpu when you want training to start."
  exit 0
fi

refuse_if_training_active

echo "=== AUTHORIZED: starting local M9a train on GPU $GPU_IDX ==="
export CUDA_VISIBLE_DEVICES="$GPU_IDX"
# Torch 2.14 native JIT routes ModernBERT RoPE through Triton, which needs
# python3.11-devel (Python.h). This host lacks those headers — disable the
# native JIT path so training uses the stock CUDA kernels instead.
export TORCH_DISABLE_NATIVE_JIT="${TORCH_DISABLE_NATIVE_JIT:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "TORCH_DISABLE_NATIVE_JIT=$TORCH_DISABLE_NATIVE_JIT"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

if (( BACKGROUND == 1 )); then
  (
    write_lock $$
    trap 'clear_lock_if_ours $$' EXIT
    set -o pipefail
    "${CMD[@]}" 2>&1 | tee -a "$LOG"
  ) &
  LAUNCHER_PID=$!
  disown "$LAUNCHER_PID" 2>/dev/null || true
  echo "Background train started (wrapper PID $LAUNCHER_PID)."
  echo "log            -> $LOG"
  echo "out (latest)   -> $OUT"
  echo "Monitor (read-only): ./training/logging/watch_m9a_run.sh --log $LOG --jsonl $OUT/train_steps.jsonl --follow"
  echo "Post-train: ./training/train/complete_run.sh --run-tag ${RUN_TAG} [--publish]"
  exit 0
fi

# Foreground train (tee to log). Trainer already fits per-head temperatures
# and writes temperatures.json + summary.json with subclass selection.
# pipefail: if train fails, do not run post-train eval on a partial artifact.
write_lock $$
trap 'clear_lock_if_ours $$' EXIT
set -o pipefail
"${CMD[@]}" 2>&1 | tee "$LOG"
echo "=== train finished; running held-out eval + calibration surfaces ==="
EVAL_JSON="reports/eval_${RUN_TAG}.json"
mkdir -p reports
CUDA_VISIBLE_DEVICES="$GPU_IDX" "$PY" training/eval/eval_modernbert.py \
  --checkpoint "$OUT" \
  --stage data/modernbert_training/stage \
  --sample 0 \
  --selective-risk \
  --write-routing-thresholds "$OUT/routing_thresholds.json" \
  --json >"$EVAL_JSON"
echo "eval report -> $EVAL_JSON"
GATE_RC=0
CUDA_VISIBLE_DEVICES="$GPU_IDX" "$PY" training/check_m9a_gates.py "$EVAL_JSON" "$OUT/summary.json" || GATE_RC=$?
echo "=== artifact map ==="
echo "  log=$LOG"
echo "  checkpoint=$OUT"
echo "  epoch_metrics=$OUT/epoch_metrics.tsv"
echo "  train_steps=$OUT/train_steps.jsonl"
echo "  eval=$EVAL_JSON"

PUBLISH_CMD=(
  "$PY" training/train/post-train/publish_run_to_hub.py
  --checkpoint "$OUT"
  --eval-json "$EVAL_JSON"
  --release-tag "$RUN_TAG"
  --invoke-check-gates
)
echo "=== Hub publish (post-train gate) ==="
echo "  model repo: Lucius-Morningstar/mailroom-modernbert-classifier (override: --repo on publish script)"
printf '  operator command: HF_TOKEN=... '
for _a in "${PUBLISH_CMD[@]}"; do printf ' %q' "$_a"; done
printf '\n'

if (( PUBLISH == 1 )); then
  if (( GATE_RC != 0 )); then
    echo "SKIP Hub publish: M9a gates not met (re-run with --ignore-gates on publish script if intentional)" >&2
    exit "$GATE_RC"
  fi
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "SKIP Hub publish: HF_TOKEN not set — dry-run plan:" >&2
    "${PUBLISH_CMD[@]}" --dry-run
    exit 0
  fi
  "${PUBLISH_CMD[@]}"
else
  echo "  (pass --publish-to-hub on this script after train to upload automatically)"
fi

exit "$GATE_RC"
