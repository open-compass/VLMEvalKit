#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
: "${EVAL_ROOT:?Set EVAL_ROOT to the evaluation output directory}"
MOCK_SERVER=${MOCK_SERVER:-.github/scripts/mock_chat_api.py}
MOCK_PORT=${MOCK_PORT:-24334}
# Match the dataset options used to generate baseline_261104.
MOCK_DATA_CONFIG=${MOCK_DATA_CONFIG:-'{
  "hle_verified": {"class": "HLEDataset", "dataset": "hle_verified", "skip_noimg": true},
  "SciDocBench": {"class": "SciDocBench", "dataset": "SciDocBench", "enable_reasoning_diagnostic": true},
  "OmniMat_QA": {"class": "OmniMat", "dataset": "OmniMat_QA", "skip_noimg": false},
  "OmniMat_CAL": {"class": "OmniMat", "dataset": "OmniMat_CAL", "skip_noimg": false}
}'}
export PYTHONPATH="$PWD"
export PRED_FORMAT=tsv
mkdir -p "$EVAL_ROOT/logs"

server_ready() {
  python - "$1" <<'PY'
import sys
import urllib.request

try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f'http://127.0.0.1:{sys.argv[1]}/health', timeout=1) as response:
        sys.exit(0 if response.status == 200 else 1)
except (OSError, urllib.error.URLError):
    sys.exit(1)
PY
}

run_type() (
  answer_type=$1
  shift
  evaluation_pid=

  cleanup_evaluation() {
    if [ -n "$evaluation_pid" ]; then
      kill "$evaluation_pid" 2>/dev/null || true
      wait "$evaluation_pid" 2>/dev/null || true
    fi
  }
  trap cleanup_evaluation EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  for dataset in "$@"; do
    echo "Starting $dataset evaluation on port $MOCK_PORT"
    python -u run.py \
      --data "$dataset" \
      --data-config "$MOCK_DATA_CONFIG" \
      --model mock/mock-api \
      --api-mode \
      --base-url "http://127.0.0.1:$MOCK_PORT/v1" \
      --extra-body "{\"mock_dataset\":\"$dataset\",\"mock_stage\":\"infer\"}" \
      --api-nproc 128 \
      --local-media \
      --judge mock-judge-api \
      --judge-base-url "http://127.0.0.1:$MOCK_PORT/v1" \
      --judge-args "{\"mock_dataset\":\"$dataset\",\"mock_stage\":\"judge\"}" \
      --judge-api-nproc 32 \
      --judge-timeout 60 \
      --judge-retry 1 \
      --work-dir "$EVAL_ROOT/mock_api_${answer_type}" \
      --reuse &
    evaluation_pid=$!
    wait "$evaluation_pid"
    evaluation_pid=
    echo "Finished $dataset evaluation"
  done
)

pids=()
server_pid=
cleanup_workers() {
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  if [ -n "$server_pid" ]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if python - "$MOCK_PORT" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(1)
    sys.exit(0 if sock.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0 else 1)
PY
then
  echo "Port $MOCK_PORT is already in use; refusing to reuse another mock server"
  exit 1
fi

python -u "$MOCK_SERVER" --port "$MOCK_PORT" > "$EVAL_ROOT/logs/mock_api.log" 2>&1 &
server_pid=$!
ready=0
for attempt in {1..60}; do
  kill -0 "$server_pid"
  if server_ready "$MOCK_PORT"; then
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo "Mock server did not become ready"
  exit 1
fi
kill -0 "$server_pid"

if [ "$#" -gt 0 ]; then
  # Explicit datasets use the same output group as the additional CI cases.
  answer_types=(extended)
  run_type extended "$@" > "$EVAL_ROOT/logs/eval_extended.log" 2>&1 &
  pids+=("$!")
else
  answer_types=(choice bool score text extended)
  run_type choice XLRS-Bench-lite MMMU_DEV_VAL MathVista_MINI AI2D_TEST MMStar \
    > "$EVAL_ROOT/logs/eval_choice.log" 2>&1 &
  pids+=("$!")
  run_type bool HallusionBench > "$EVAL_ROOT/logs/eval_bool.log" 2>&1 &
  pids+=("$!")
  run_type score MMVet > "$EVAL_ROOT/logs/eval_score.log" 2>&1 &
  pids+=("$!")
  run_type text OCRBench > "$EVAL_ROOT/logs/eval_text.log" 2>&1 &
  pids+=("$!")
  run_type extended hle_verified MMMU_Pro_10c ChartQAPro SciDocBench MicroVQA SFE OmniMat_QA OmniMat_CAL \
    > "$EVAL_ROOT/logs/eval_extended.log" 2>&1 &
  pids+=("$!")
fi

result=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    result=1
  fi
  unset 'pids[index]'
done

python - "$MOCK_PORT" > "$EVAL_ROOT/logs/mock_requests.json" <<'PY'
import json
import sys
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(f'http://127.0.0.1:{sys.argv[1]}/health', timeout=5) as response:
    print(json.dumps(json.load(response), indent=2))
PY

tail -n 20 "$EVAL_ROOT/logs/mock_api.log" || true
for answer_type in "${answer_types[@]}"; do
  echo "=== $answer_type ==="
  tail -n 20 "$EVAL_ROOT/logs/eval_${answer_type}.log" || true
done
exit "$result"
