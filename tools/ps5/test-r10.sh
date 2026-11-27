#!/usr/bin/env bash
# One prompt per process; six arms. Replay covers prefill only. Not a benchmark.
set -euo pipefail
[[ $# == 2 && -r $1 ]] || { echo 'usage: test-r10.sh GENERIC-MOE.gguf NEW_OUTPUT_DIR' >&2; exit 2; }
model=$(realpath "$1"); out=$(realpath -m "$2"); mkdir "$out"
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
while IFS= read -r name; do unset "$name"; done < <(compgen -A variable STRATA_)
export STRATA_MODEL="$model" STRATA_PS5=1 STRATA_PS5_GTT=1 STRATA_PS5_DIRECT=1 STRATA_TEST_BACKENDS=vulkan
export STRATA_KV=q8 STRATA_KV_K=q8 STRATA_KV_V=q8
export STRATA_PS5_GEMV_WG=256 STRATA_PS5_SMALL_GEMM=0 STRATA_PS5_ATTN_WG=0 STRATA_PS5_ZERO_COPY=0 STRATA_PS5_R7_GEMM_TILE=0 STRATA_VK_NO_COOP=1
export STRATA_PS5_R4_RESEARCH=1 STRATA_PS5_SCHED_SCRATCH_MB=64 STRATA_PS5_SCHED_MAX_ROWS=32 STRATA_FIXED_EXPERTS=1 STRATA_EXPERT_MB=0 RAYON_NUM_THREADS=6 STRATA_PS5_SCHED=gpu
export STRATA_PS5_R9_LAYER="${PS5_R9_LAYER:-0}" STRATA_PS5_R9_POS="${PS5_R9_POS:-0}" STRATA_PS5_R9_TOKEN="${PS5_R9_TOKEN:-0}" STRATA_PS5_R9_RANK="${PS5_R9_RANK:-0}"
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
git rev-parse HEAD > "$out/commit.txt"
cargo test "${args[@]}" --lib ps5_r10::tests 2>&1 | tee "$out/unit.log"
cargo test "${args[@]}" --test ps5_r4 --no-run --message-format=json-render-diagnostics > "$out/build.jsonl"
bin=$(python3 - "$out/build.jsonl" <<'PY'
import json,sys
p=[r['executable'] for s in open(sys.argv[1]) if (r:=json.loads(s)).get('target',{}).get('name')=='ps5_r4' and r.get('executable')]
assert len(p)==1,p
print(p[0])
PY
)
sha256sum "$bin" > "$out/binary.sha256"
for prompt in 0 1; do
  d="$out/prompt-$prompt"; mkdir "$d"
  for arm in gpu shadow sham replay-shadow contract replay-contract; do
    mode=record; probe=1; r9=$arm
    case "$arm" in gpu) mode=; probe=0; r9=;; replay-*) mode=replay; r9=${arm#replay-};; esac
    iq=0
    if [[ ${PS5_R11_CONTRACT:-0} == 1 ]]; then iq=1; fi
    boundary=0
    if [[ ${PS5_R12_BOUNDARY_TRACE:-0} == 1 && $probe == 1 ]]; then boundary=1; fi
    STRATA_PS5_R12_BOUNDARY_TRACE="$boundary" STRATA_PS5_R11_IQ_F16="$iq" STRATA_PS5_R11_IQ_Q32="$iq" STRATA_PS5_R11_LOG="$iq" \
    PS5_R10_PROMPT="$prompt" STRATA_PS5_R9_MODE="$r9" STRATA_R7_R4_STAGE_PROBE="$probe" \
      STRATA_PS5_R10_ROUTES="$mode" STRATA_PS5_R10_ROUTE_FILE="$d/routes.json" PS5_R4_PARITY_FILE="$d/$arm.json" \
      python3 tools/ps5/r8-run-guarded.py --out "$d/$arm" --timeout "${PS5_R10_TIMEOUT:-5400}" --min-kib 1048576 -- \
      "$bin" prefill_scheduler_parity --ignored --exact --nocapture --test-threads=1 > "$d/$arm.log" 2>&1
    if [[ $arm == shadow ]]; then python3 tools/ps5/compare-r10.py tape "$d"; fi
  done
  if [[ ${PS5_R11_CONTRACT:-0} == 1 ]]; then
    for arm in shadow sham replay-shadow contract replay-contract; do grep -Fq "R11 IQ exercised:" "$d/$arm.log"; done
  fi
  python3 tools/ps5/compare-r10.py compare "$d"
  if [[ ${PS5_R12_BOUNDARY_TRACE:-0} == 1 ]]; then python3 tools/ps5/compare-r12-boundaries.py "$d"; fi
done
