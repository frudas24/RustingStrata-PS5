#!/usr/bin/env bash
# Use only after test-r4.sh passes. The off arm preserves R3; gpu isolates synchronization overhead.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
[[ $# -ge 1 && -r $1 ]] || { echo 'usage: bench-r4.sh MODEL.gguf [OUTPUT_DIR] [extra bench flags]' >&2; exit 2; }
model=$(realpath "$1"); shift; out=${1:-ps5-r4-$(date +%Y%m%d-%H%M%S)}; if (($#)); then shift; fi
mkdir -p "$out"
export STRATA_PS5=1 STRATA_PS5_GTT=1 STRATA_PS5_DIRECT=1 STRATA_FIXED_EXPERTS=1 STRATA_PS5_ZERO_COPY=0
export STRATA_PS5_GEMV_WG=${STRATA_PS5_GEMV_WG:-256} STRATA_PS5_SMALL_GEMM=${STRATA_PS5_SMALL_GEMM:-0}
export STRATA_EXPERT_MB=${STRATA_EXPERT_MB:-512} STRATA_PS5_SCHED_LOG=1
export STRATA_PS5_SCHED_MAX_ROWS=${STRATA_PS5_SCHED_MAX_ROWS:-4} STRATA_PS5_SCHED_SCRATCH_MB=${STRATA_PS5_SCHED_SCRATCH_MB:-64}
export PS5_CTX=${PS5_CTX:-8192} PS5_CHUNK=${PS5_CHUNK:-128} PS5_KV=${PS5_KV:-q8} PS5_THREADS=${PS5_THREADS:-6}
args=(bench --model "$model" --pp "${PS5_PP:-128,512,2048}" --tg "${PS5_TG:-32}" --reps "${PS5_REPS:-5}" --mtp 0 "$@")
"$here/diagnose.sh" > "$out/diagnostics.txt" 2>&1 || true
printf '%s\n' "expert_mb=$STRATA_EXPERT_MB" "max_rows=$STRATA_PS5_SCHED_MAX_ROWS" "scratch_mb=$STRATA_PS5_SCHED_SCRATCH_MB" "chunk=$PS5_CHUNK" "threads=$PS5_THREADS" > "$out/config.txt"
for case in off1 gpu half auto1 auto2 off2; do
  mode=${case%[12]}
  rm -f "$out/$case.json"
  STRATA_PS5_SCHED=$mode "$here/run.sh" "${args[@]}" --json-out "$out/$case.json" 2>&1 | tee "$out/$case.log"
  test -s "$out/$case.json"
done
python3 "$here/summarize-r4.py" "$out"
