#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
: "${EVAL_ROOT:?Set EVAL_ROOT to the evaluation output directory}"
MOCK_SERVER=${MOCK_SERVER:-/mnt/shared-storage-user/opencompass-shared/qa-llm-cicd/mock_chat_api.py}
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

run_type() {
  answer_type=$1
  port=$2
  shift 2
  evaluation_pid=

  if python - "$port" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(1)
    sys.exit(0 if sock.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0 else 1)
PY
  then
    echo "Port $port is already in use; refusing to reuse another mock server"
    exit 1
  fi

  python "$MOCK_SERVER" --type "$answer_type" --port "$port" \
    > "$EVAL_ROOT/logs/mock_${answer_type}.log" 2>&1 &
  server_pid=$!

  cleanup() {
    if [ -n "$evaluation_pid" ]; then
      kill "$evaluation_pid" 2>/dev/null || true
      wait "$evaluation_pid" 2>/dev/null || true
    fi
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  }
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  ready=0
  for attempt in {1..60}; do
    kill -0 "$server_pid"
    if server_ready "$port"; then
      ready=1
      break
    fi
    sleep 1
  done
  if [ "$ready" -ne 1 ]; then
    echo "Mock server for $answer_type did not become ready"
    exit 1
  fi
  kill -0 "$server_pid"

  echo "Starting $answer_type evaluation on port $port"
  python -u run.py \
    --data "$@" \
    --model mock/mock-api \
    --api-mode \
    --base-url "http://127.0.0.1:$port/v1" \
    --api-nproc 128 \
    --local-media \
    --judge mock-judge-api \
    --judge-base-url "http://127.0.0.1:$port/v1" \
    --judge-api-nproc 32 \
    --judge-timeout 60 \
    --judge-retry 1 \
    --work-dir "$EVAL_ROOT/mock_api_${answer_type}" \
    --reuse &
  evaluation_pid=$!
  wait "$evaluation_pid"
  evaluation_pid=
  echo "Finished $answer_type evaluation"
}

pids=()
cleanup_workers() {
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_type choice 24334 XLRS-Bench-lite MMMU_DEV_VAL MathVista_MINI AI2D_TEST MMStar \
  > "$EVAL_ROOT/logs/eval_choice.log" 2>&1 &
pids+=("$!")
run_type bool 24335 HallusionBench > "$EVAL_ROOT/logs/eval_bool.log" 2>&1 &
pids+=("$!")
run_type score 24336 MMVet > "$EVAL_ROOT/logs/eval_score.log" 2>&1 &
pids+=("$!")
run_type text 24337 OCRBench > "$EVAL_ROOT/logs/eval_text.log" 2>&1 &
pids+=("$!")

result=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    result=1
  fi
  unset 'pids[index]'
done

for answer_type in choice bool score text; do
  echo "=== $answer_type ==="
  tail -n 20 "$EVAL_ROOT/logs/mock_${answer_type}.log" "$EVAL_ROOT/logs/eval_${answer_type}.log" || true
done
exit "$result"
