#!/usr/bin/env bash
#
# rbln_smoke.sh — Tier 2 per-family RBLN inference smoke test.
#
# Runs one representative model per RBLN wrapper family through
# `run.py --device rbln` on a small dataset slice, exercising the real
# compile-or-load seam in vlmeval/vlm/rbln/base.py. This is the
# primary deliverable: proof that every wrapper family produces real
# NPU output. It is NOT a CI check — run it on an NPU host.
#
# Design (per plan):
#   * tensor_parallel_size is a COMPILE-TIME choice, not a static device
#     gate. If a compiled artifact (./<basename>/*.rbln) already exists it
#     is loaded as-is; otherwise we compile with the _ARCH_TABLE default
#     tp, or a per-family override.
#   * A *free* pre-compile sanity check skips a family only when its
#     effective tp provably exceeds the visible NPU count (turns a long
#     compile-then-fail into an instant skip). This is not the rejected
#     static matrix gate — it is override-aware and bypassable.
#   * Stale-cache trap: if a cached artifact exists AND a tp override is
#     requested, the override is silently ignored on load (base.py drops
#     compile-time rbln_config when export=False). We WARN loudly and
#     point at `rm -rf` as the durable remedy.
#   * Per-family cache isolation via cd, because _save_dir keys on
#     basename relative to CWD (independent of --work-dir).
#
# Usage:
#   bash scripts/rbln_smoke.sh                 # all families
#   bash scripts/rbln_smoke.sh qwen2_vl        # single family
#   RBLN_TP_qwen2_vl=4 bash scripts/rbln_smoke.sh qwen2_vl   # tp override
#   DRY_RUN=1 bash scripts/rbln_smoke.sh       # print actions, no NPU
#
# Env:
#   LIMIT       (default 2)   samples per dataset
#   WORK_ROOT   (default ./outputs/rbln_smoke)
#   CACHE_ROOT  (default ./outputs/rbln_smoke/_cache) per-family cwd root
#   RBLN_TP_<family>          override compile tensor_parallel_size
#   DRY_RUN=1                 print run.py invocations instead of executing

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIMIT="${LIMIT:-2}"
WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/outputs/rbln_smoke}"
CACHE_ROOT="${CACHE_ROOT:-${WORK_ROOT}/_cache}"
PY="${PYTHON:-python}"

FAMILIES=(
  qwen2_vl qwen3_vl cosmos llava llava_next
  idefics3 gemma3 pixtral paligemma paligemma2 blip2
)

# family -> HF model id (basename becomes the cache dir)
declare -A MODEL=(
  [qwen2_vl]="Qwen/Qwen2.5-VL-7B-Instruct"
  [qwen3_vl]="Qwen/Qwen3-VL-8B-Instruct"
  [cosmos]="nvidia/Cosmos-Reason1-7B"
  [llava]="llava-hf/llava-1.5-7b-hf"
  [llava_next]="llava-hf/llava-v1.6-mistral-7b-hf"
  [idefics3]="HuggingFaceM4/Idefics3-8B-Llama3"
  [gemma3]="google/gemma-3-4b-it"
  [pixtral]="mistral-community/pixtral-12b"
  [paligemma]="google/paligemma-3b-mix-448"
  [paligemma2]="google/paligemma2-3b-mix-224"
  [blip2]="Salesforce/blip2-opt-2.7b"
)

# family -> representative dataset (small MCQ/VQA; cached or auto-download)
declare -A DATA=(
  [qwen2_vl]="MMBench_DEV_EN"
  [qwen3_vl]="MMStar"
  [cosmos]="MMBench_DEV_EN"
  [llava]="MMStar"
  [llava_next]="MMBench_DEV_EN"
  [idefics3]="MMStar"
  [gemma3]="AI2D_TEST"
  [pixtral]="MMBench_DEV_EN"
  [paligemma]="AI2D_TEST"
  [paligemma2]="AI2D_TEST"
  [blip2]="ChartQA_TEST"
)

# family -> _ARCH_TABLE default tensor_parallel_size
declare -A DEF_TP=(
  [qwen2_vl]=8 [qwen3_vl]=8 [cosmos]=8
  [llava]=4 [llava_next]=4 [idefics3]=8 [gemma3]=4
  [pixtral]=8 [paligemma]=4 [paligemma2]=4 [blip2]=1
)

# family -> printf template that injects a tp value into --rbln-kwargs JSON.
# The nested key path differs per family (see _ARCH_TABLE).
declare -A TMPL=(
  [qwen2_vl]='{"rbln_config":{"tensor_parallel_size":%d,"visual":{"max_seq_lens":6400}}}'
  [qwen3_vl]='{"rbln_config":{"tensor_parallel_size":%d,"visual":{"max_seq_lens":16384}}}'
  [cosmos]='{"rbln_config":{"tensor_parallel_size":%d,"visual":{"max_seq_lens":8192}}}'
  [llava]='{"rbln_config":{"language_model":{"tensor_parallel_size":%d}}}'
  [llava_next]='{"rbln_config":{"language_model":{"tensor_parallel_size":%d}}}'
  [idefics3]='{"rbln_config":{"text_model":{"tensor_parallel_size":%d}}}'
  [gemma3]='{"rbln_config":{"language_model":{"tensor_parallel_size":%d}}}'
  [pixtral]='{"rbln_config":{"language_model":{"tensor_parallel_size":%d}}}'
  [paligemma]='{"rbln_config":{"language_model":{"tensor_parallel_size":%d}}}'
  [paligemma2]='{"rbln_config":{"language_model":{"tensor_parallel_size":%d}}}'
  [blip2]='{"rbln_config":{"language_model":{"tensor_parallel_size":%d}}}'
)

log()  { printf '[rbln-smoke] %s\n' "$*"; }
warn() { printf '[rbln-smoke][WARN] %s\n' "$*" >&2; }
err()  { printf '[rbln-smoke][ERROR] %s\n' "$*" >&2; }

npu_count() {
  # RBLN NPUs are exposed as /dev/rbln0../dev/rblnN (NOT /dev/rsd*, which is
  # an unrelated system device). Prefer `rbln-stat` when available, falling
  # back to a /dev/rbln* count.
  local n
  n=$(find /dev -maxdepth 1 -name 'rbln*' 2>/dev/null | wc -l)
  printf '%s' "$n"
}

has_compiled_artifact() {
  # $1 = cache dir
  [ -d "$1" ] && find "$1" -maxdepth 1 -name '*.rbln' 2>/dev/null | grep -q .
}

run_family() {
  local fam="$1"
  local model="${MODEL[$fam]}"
  local data="${DATA[$fam]}"
  local def_tp="${DEF_TP[$fam]}"

  # Effective compile tp: per-family env override, else _ARCH_TABLE default.
  local tp_var="RBLN_TP_${fam}"
  local eff_tp="${!tp_var:-$def_tp}"
  local override_set=0
  [ "$eff_tp" != "$def_tp" ] && override_set=1

  local npus
  npus="$(npu_count)"

  # Free pre-compile sanity check (NOT a static matrix gate): only refuse a
  # compile that provably cannot place. Bypassable via a smaller tp override.
  # Applied in DRY_RUN too so the gate's behaviour is verifiable without NPU.
  if [ "$npus" -gt 0 ] && [ "$eff_tp" -gt "$npus" ]; then
    warn "BLOCKED-PRE-COMPILE: ${fam} needs tp=${eff_tp} but only ${npus} NPU(s) visible."
    warn "  -> set RBLN_TP_${fam}=<=${npus} to compile at a smaller tp, or run on a larger board."
    return 2
  fi

  # Per-family cache isolation: cd into a dedicated dir so _save_dir's
  # basename-relative cache cannot collide across families.
  local fam_cwd="${CACHE_ROOT}/${fam}"
  mkdir -p "$fam_cwd"
  local basename_dir
  basename_dir="${model##*/}"
  local cache_dir="${fam_cwd}/${basename_dir}"

  # Stale-cache trap: a cached artifact ignores compile-time --rbln-kwargs.
  local rbln_kwargs=""
  if [ "$override_set" -eq 1 ]; then
    if has_compiled_artifact "$cache_dir"; then
      warn "${fam}: cached artifact at ${cache_dir} — tp override (${eff_tp}) will be IGNORED on load."
      warn "  -> rm -rf '${cache_dir}' to recompile at tp=${eff_tp}."
    fi
    # shellcheck disable=SC2059
    rbln_kwargs="$(printf "${TMPL[$fam]}" "$eff_tp")"
  fi

  local work_dir="${WORK_ROOT}/${fam}"
  mkdir -p "$work_dir"

  local -a cmd=(
    "$PY" "${REPO_ROOT}/run.py"
    --device rbln
    --model "$model"
    --data "$data"
    --limit "$LIMIT"
    --work-dir "$work_dir"
    --verbose
  )
  [ -n "$rbln_kwargs" ] && cmd+=(--rbln-kwargs "$rbln_kwargs")

  log "${fam}: model=${model} data=${data} eff_tp=${eff_tp} npus=${npus} cwd=${fam_cwd}"
  if has_compiled_artifact "$cache_dir"; then
    log "  cache HIT (${cache_dir}) -> natural load (export=False)"
  else
    log "  cache MISS -> compile (tp=${eff_tp}), artifact will be saved to ${cache_dir}"
  fi

  if [ "$DRY_RUN_FLAG" -eq 1 ]; then
    log "  DRY_RUN: (cd '${fam_cwd}' && ${cmd[*]})"
    return 0
  fi

  ( cd "$fam_cwd" && "${cmd[@]}" )
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    err "${fam}: run.py exited ${rc} (possible BLOCKED-AT-RUNTIME if tp>NPU). See log above."
    return 1
  fi

  # Assert a non-empty prediction file was produced under work_dir.
  local pred
  pred="$(find "$work_dir" -type f \( -name '*.xlsx' -o -name '*.pkl' -o -name '*.tsv' \) -newermt '-1 hour' 2>/dev/null | head -n1)"
  if [ -z "$pred" ] || [ ! -s "$pred" ]; then
    err "${fam}: no non-empty prediction file found under ${work_dir}"
    return 1
  fi
  log "  OK: prediction file ${pred}"
  return 0
}

main() {
  DRY_RUN_FLAG="${DRY_RUN:-0}"
  [ "$DRY_RUN_FLAG" = "1" ] && DRY_RUN_FLAG=1 || DRY_RUN_FLAG=0

  local targets=()
  if [ "$#" -ge 1 ]; then
    targets=("$@")
  else
    targets=("${FAMILIES[@]}")
  fi

  local pass=0 fail=0 skip=0
  for fam in "${targets[@]}"; do
    if [ -z "${MODEL[$fam]:-}" ]; then
      err "unknown family: ${fam} (known: ${FAMILIES[*]})"
      fail=$((fail + 1))
      continue
    fi
    run_family "$fam"
    case $? in
      0) pass=$((pass + 1)) ;;
      2) skip=$((skip + 1)) ;;
      *) fail=$((fail + 1)) ;;
    esac
  done

  log "summary: pass=${pass} skip=${skip} fail=${fail}"
  [ "$fail" -eq 0 ]
}

main "$@"
