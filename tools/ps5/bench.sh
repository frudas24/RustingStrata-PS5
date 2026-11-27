#!/usr/bin/env bash
# Keep the same model, context, chunk, KV, seed/greedy and threads in every mode.
set -euo pipefail
export STRATA_PS5_ZERO_COPY=0
here=$(cd -- "$(dirname -- "$0")" && pwd)
[[ $# -ge 1 ]] || { echo 'usage: bench.sh MODEL.gguf [OUTPUT_DIR] [additional bench flags...]' >&2; exit 2; }
model=$1; shift; [[ -r "$model" ]] || { echo "Model not readable: $model" >&2; exit 2; }
out=${1:-ps5-bench-$(date +%Y%m%d-%H%M%S)}; if (($#)); then shift; fi
mkdir -p "$out"
"$here/diagnose.sh" > "$out/diagnostics.txt" 2>&1 || true
# Fix expert placement profile updates during A/B; use the same initial layer-order placement.
export STRATA_FIXED_EXPERTS=1
# All cases use explicit same settings; optional old binary is the actual upstream baseline.
export PS5_CTX=${PS5_CTX:-8192} PS5_CHUNK=${PS5_CHUNK:-512} PS5_KV=${PS5_KV:-q8} PS5_THREADS=${PS5_THREADS:-6}
base_args=(bench --model "$model" --pp "${PS5_PP:-512,2048,4096}" --tg "${PS5_TG:-128}" --reps "${PS5_REPS:-3}" --mtp 0 "$@")
run_case() {
  local name=$1 profile=$2 gtt=$3 direct=$4; shift 4
  echo "=== $name ==="
  rm -f -- "$out/$name.json"
  if STRATA_PS5=$profile STRATA_PS5_GTT=$gtt STRATA_PS5_DIRECT=$direct "$@" "$here/run.sh" "${base_args[@]}" --json-out "$out/$name.json" 2>&1 | tee "$out/$name.log"; then :; else echo "$name failed" >&2; fi
}
if [[ -n ${BASELINE_BIN:-} ]]; then run_case upstream 0 0 0 env STRATA_PS5_BIN="$BASELINE_BIN"; fi
run_case control 0 0 0 env
run_case guarded 1 0 0 env
run_case gtt 1 1 0 env
run_case mapped 1 0 1 env
run_case combined 1 1 1 env
python3 "$here/summarize.py" "$out"
