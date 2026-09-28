#!/usr/bin/env bash
# Copy generated ModernBERT reports into a local mailroom-sandbox checkout.
#
# Usage:
#   ./training/sync_reports_to_sandbox.sh \
#     --sandbox-root ../mailroom-sandbox \
#     --run-tag m9a-local-20260927-014429 \
#     --run-number 03
#
# Requires mailroom-ml reports already generated via write_eval_report.py.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SANDBOX=""
RUN_TAG=""
RUN_NUM=""

usage() {
  sed -n '2,10p' "$0"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sandbox-root=*) SANDBOX="${1#*=}" ;;
    --sandbox-root) SANDBOX="${2:-}"; shift ;;
    --run-tag=*) RUN_TAG="${1#*=}" ;;
    --run-tag) RUN_TAG="${2:-}"; shift ;;
    --run-number=*) RUN_NUM="${1#*=}" ;;
    --run-number) RUN_NUM="${2:-}"; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown: $1" >&2; usage 1 ;;
  esac
  shift
done

if [[ -z "$SANDBOX" || -z "$RUN_TAG" || -z "$RUN_NUM" ]]; then
  echo "ERROR: --sandbox-root, --run-tag, --run-number required" >&2
  usage 1
fi

DEST="${SANDBOX}/reports/modernbert"
mkdir -p "$DEST"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

EVAL="${ROOT}/reports/eval_${RUN_TAG}.json"
HELDOUT="${ROOT}/reports/eval_${RUN_TAG}-heldout-plus.json"
TEST_MD="${ROOT}/reports/TEST-EVAL/TEST-EVAL-REPORT-${RUN_TAG}.md"
TRAIN_MD="${ROOT}/reports/M9a-REPORT-${RUN_TAG}.md"

[[ -f "$EVAL" ]] && cp "$EVAL" "${DEST}/eval_${RUN_TAG}.json"
[[ -f "$HELDOUT" ]] && cp "$HELDOUT" "${DEST}/eval_${RUN_TAG}-heldout-plus.json"
[[ -f "$TEST_MD" ]] && cp "$TEST_MD" "${DEST}/RUN-${RUN_NUM}-MODERNBERT-HELDOUT-TEST-REPORT.md"
[[ -f "$TRAIN_MD" ]] && cp "$TRAIN_MD" "${DEST}/RUN-${RUN_NUM}-MODERNBERT-TRAINING-REPORT.md"

README="${DEST}/README.md"
{
  echo "# ModernBERT reports (synced from mailroom-ml)"
  echo ""
  echo "Generated from [\`mailroom-ml\`](https://github.com/LLM-Mailroom-Services/mailroom-ml)"
  echo "at \`${SHA}\` (${STAMP} UTC) via \`training/sync_reports_to_sandbox.sh\`."
  echo ""
  echo "| run_tag | held-out test | training | eval JSON |"
  echo "| --- | --- | --- | --- |"
  echo "| \`${RUN_TAG}\` | RUN-${RUN_NUM}-MODERNBERT-HELDOUT-TEST-REPORT.md | RUN-${RUN_NUM}-MODERNBERT-TRAINING-REPORT.md | eval_${RUN_TAG}.json |"
  echo ""
  echo "Canonical authoring lives in mailroom-ml \`reports/TEST-EVAL/\`; this tree is a mirror."
} > "$README"

echo "synced to ${DEST}"
