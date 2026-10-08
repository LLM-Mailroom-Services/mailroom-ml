#!/usr/bin/env bash
# Read-only M9a training monitor — polls logs and process list; never starts or kills jobs.
#
# Usage:
#   ./training/logging/watch_m9a_run.sh
#   ./training/logging/watch_m9a_run.sh --once
#   ./training/logging/watch_m9a_run.sh --log logs/m9a-local-....log --jsonl data/.../train_steps.jsonl
#   ./training/logging/watch_m9a_run.sh --follow   # refresh every 30s until Ctrl-C
#   ./training/logging/watch_m9a_run.sh --agent    # sparse follow every 42m (agents / long ETA)
#   ./training/logging/watch_m9a_run.sh --follow --pretty   # LLM Mailroom ASCII TUI (default on TTY)
#   ./training/logging/watch_m9a_run.sh --plain   # force plain text
#
# Safe for agents/subagents: use this instead of re-launching run_m9a_local.sh when
# progress looks slow. If train_modernbert is running, only watch.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
LOCK="${ROOT}/logs/.m9a-train.lock"
LOG=""
JSONL=""
EPOCH_JSONL=""
ONCE=0
FOLLOW=0
INTERVAL=30
INTERVAL_SET=0
# Long-run agent/automation polls (42 minutes) — avoid burning tokens on multi-hour training.
AGENT_INTERVAL=2520
PRETTY=""
PLAIN=0

usage() {
  sed -n '2,12p' "$0"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log=*) LOG="${1#*=}" ;;
    --log) LOG="${2:-}"; shift ;;
    --jsonl=*) JSONL="${1#*=}" ;;
    --jsonl) JSONL="${2:-}"; shift ;;
    --once) ONCE=1 ;;
    --follow) FOLLOW=1 ;;
    --agent) FOLLOW=1; INTERVAL="$AGENT_INTERVAL" ;;
    --interval=*) INTERVAL="${1#*=}"; INTERVAL_SET=1 ;;
    --pretty) PRETTY=1 ;;
    --plain) PLAIN=1 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
  shift
done

if [[ -z "$LOG" && -f "$LOCK" ]]; then
  # lock line: <pid> <run_tag> <log_path> [optional fields...]
  read -r _ _ maybe_log _ < "$LOCK" || true
  if [[ -n "${maybe_log:-}" && -f "$ROOT/$maybe_log" ]]; then
    LOG="$maybe_log"
  elif [[ -n "${maybe_log:-}" && -f "$maybe_log" ]]; then
    LOG="$maybe_log"
  fi
fi

if [[ -z "$JSONL" && -n "$LOG" ]]; then
  tag="$(basename "$LOG" .log)"
  guess="$ROOT/data/modernbert_training/runs/${tag}/latest/train_steps.jsonl"
  if [[ -f "$guess" ]]; then
    JSONL="${guess#$ROOT/}"
  fi
fi

if [[ -z "$EPOCH_JSONL" && -n "$LOG" ]]; then
  tag="$(basename "$LOG" .log)"
  guess_epoch="$ROOT/data/modernbert_training/runs/${tag}/latest/epoch_metrics.jsonl"
  if [[ -f "$guess_epoch" ]]; then
    EPOCH_JSONL="${guess_epoch#$ROOT/}"
  fi
fi

if [[ -z "$PRETTY" ]]; then
  if (( PLAIN == 0 )) && [[ -t 1 ]]; then
    PRETTY=1
  else
    PRETTY=0
  fi
fi

print_status() {
  local ts lock_line run_tag trainer_lines logpath jpath ejpath pretty_args
  ts="$(date '+%Y-%m-%d %H:%M:%S')"

  lock_line=""
  run_tag=""
  # Explicit --log wins for title/metrics; lock is still shown for the canonical launcher.
  if [[ -n "$LOG" ]]; then
    run_tag="$(basename "$LOG" .log)"
  fi
  if [[ -f "$LOCK" ]]; then
    lock_line="$(cat "$LOCK")"
    if [[ -z "$LOG" ]]; then
      read -r _ run_tag _ <<< "$lock_line" || true
    fi
  fi

  trainer_lines=""
  for pid in $(pgrep -f 'train_modernbert\.py' 2>/dev/null || true); do
    args="$(ps -o args= -p "$pid" 2>/dev/null || true)"
    if [[ "$args" == *python*train_modernbert.py* ]]; then
      if [[ -n "$run_tag" && "$args" != *"runs/${run_tag}/"* ]]; then
        continue
      fi
      trainer_lines+="${pid} ${args}"$'\n'
    fi
  done
  trainer_lines="${trainer_lines%$'\n'}"

  logpath=""
  if [[ -n "$LOG" ]]; then
    logpath="$LOG"
    [[ "$logpath" != /* ]] && logpath="$ROOT/$logpath"
  fi

  jpath=""
  if [[ -n "$JSONL" ]]; then
    jpath="$JSONL"
    [[ "$jpath" != /* ]] && jpath="$ROOT/$jpath"
  fi

  ejpath=""
  if [[ -n "$EPOCH_JSONL" ]]; then
    ejpath="$EPOCH_JSONL"
    [[ "$ejpath" != /* ]] && ejpath="$ROOT/$ejpath"
  fi

  if (( PRETTY == 1 )) && [[ -x "$PY" ]]; then
    pretty_args=( -m training.logging.pretty_log --once --timestamp "$ts" )
    (( PLAIN == 1 )) && pretty_args+=( --plain )
    [[ -n "$run_tag" ]] && pretty_args+=( --run-tag "$run_tag" )
    [[ -n "$lock_line" ]] && pretty_args+=( --lock-line "$lock_line" )
    if [[ -n "$trainer_lines" ]]; then
      while IFS= read -r tline; do
        [[ -n "$tline" ]] && pretty_args+=( --trainer-line "$tline" )
      done <<< "$trainer_lines"
    fi
    [[ -n "$jpath" && -f "$jpath" ]] && pretty_args+=( --jsonl "$jpath" )
    [[ -n "$ejpath" && -f "$ejpath" ]] && pretty_args+=( --epoch-jsonl "$ejpath" )
    [[ -n "$logpath" && -f "$logpath" ]] && pretty_args+=( --log "$logpath" )
    if ( cd "$ROOT" && "$PY" "${pretty_args[@]}" ) 2>/dev/null; then
      return
    fi
  fi

  echo "=== M9a watch @ $ts ==="
  if [[ -n "$lock_line" ]]; then
    echo "lock: $lock_line"
  else
    echo "lock: (none)"
  fi
  if [[ -n "$trainer_lines" ]]; then
    echo "trainer:"
    echo "$trainer_lines" | sed 's/^/  /'
  else
    echo "trainer: (not running)"
  fi
  if [[ -n "$logpath" && -f "$logpath" ]]; then
    echo "log ($LOG) last lines:"
    tail -n 3 "$logpath" | sed 's/^/  /'
  elif [[ -n "$LOG" ]]; then
    echo "log: missing $LOG"
  fi
  if [[ -n "$jpath" && -f "$jpath" ]]; then
    echo "train_steps.jsonl last event:"
    tail -n 1 "$jpath" | sed 's/^/  /'
  elif [[ -n "$JSONL" ]]; then
    echo "jsonl: missing $JSONL"
  fi
  echo
}

print_status
if (( ONCE == 1 )); then
  exit 0
fi

if (( FOLLOW == 0 )); then
  echo "Tip: --follow for periodic refresh; this script never launches training."
  exit 0
fi

# Pretty follow redraws one frame in the alternate screen. The 30s poll
# appends a new panel each time, so older art stays in scrollback and looks
# like the display never updated.
if (( PRETTY == 1 )) && [[ -x "$PY" ]]; then
  live_iv="$INTERVAL"
  (( INTERVAL_SET == 0 )) && live_iv=2
  live_args=( -m training.logging.pretty_log --follow --interval "$live_iv" )
  (( PLAIN == 1 )) && live_args+=( --plain )
  follow_tag=""
  if [[ -n "$LOG" ]]; then
    live_args+=( --log "$LOG" )
    follow_tag="$(basename "$LOG" .log)"
    live_args+=( --run-tag "$follow_tag" )
  fi
  [[ -n "$JSONL" ]] && live_args+=( --jsonl "$JSONL" )
  [[ -n "$EPOCH_JSONL" ]] && live_args+=( --epoch-jsonl "$EPOCH_JSONL" )
  # Only attach canonical lock when watching that run (avoid GPU0 lock on GPU1 log watch).
  if [[ -f "$LOCK" ]]; then
    lock_tag=""
    read -r _ lock_tag _ < "$LOCK" 2>/dev/null || lock_tag=""
    if [[ -z "$follow_tag" || "$follow_tag" == "$lock_tag" ]]; then
      live_args+=( --lock-path "$LOCK" )
    fi
  fi
  if ( cd "$ROOT" && "$PY" "${live_args[@]}" ); then
    exit 0
  fi
fi

while true; do
  sleep "$INTERVAL"
  print_status
done
