#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: bash $0 CURRENT_EVAL_ROOT [BASELINE_ROOT]" >&2
  exit 2
fi

current_root=$1
baseline_root=${2:-/mnt/shared-storage-user/mllm/qa-llm-cicd/eval_report/baseline_261104}
result=0

for answer_type in choice bool score text extended; do
  echo "Comparing mock_api_${answer_type}"
  if ! python "$(dirname "$0")/compare_dirs.py" \
    "$baseline_root/mock_api_${answer_type}/mock--mock-api" \
    "$current_root/mock_api_${answer_type}/mock--mock-api"; then
    result=1
  fi
done

exit "$result"
