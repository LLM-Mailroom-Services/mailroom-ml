#!/usr/bin/env bash
# Publish a local M9a (or other) train run to the Hub model repo.
#
# Requires HF_TOKEN (or HUGGING_FACE_HUB_TOKEN). Without a token, behaves like
# --dry-run (prints plan only).
#
# Usage:
#   ./training/train/post-train/publish_run_to_hub.sh --dry-run \\
#     --checkpoint data/modernbert_training/runs/m9a-local-.../latest \\
#     --eval-json reports/eval_m9a-local-....json
#
#   HF_TOKEN=... ./training/train/post-train/publish_run_to_hub.sh \\
#     --checkpoint .../latest --eval-json reports/eval_....json \\
#     --release-tag m9a-local-20260927-010430

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3
exec "$PY" training/train/post-train/publish_run_to_hub.py --invoke-check-gates "$@"
