#!/usr/bin/env bash
# Run only after test-r3.sh succeeds. Compare same R2 build and same kernels.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
[[ $# -ge 1 && -r $1 ]] || { echo 'usage: bench-r3.sh MODEL.gguf [OUTPUT_DIR] [extra bench flags]' >&2; exit 2; }
model=$1; shift; out=${1:-ps5-r3-$(date +%Y%m%d-%H%M%S)}; if (($#)); then shift; fi
mkdir -p "$out"
export STRATA_PS5=1 STRATA_PS5_GTT=1 STRATA_PS5_DIRECT=1 STRATA_FIXED_EXPERTS=1
export STRATA_PS5_GEMV_WG=${STRATA_PS5_GEMV_WG:-256} STRATA_PS5_SMALL_GEMM=${STRATA_PS5_SMALL_GEMM:-0}
export PS5_CTX=${PS5_CTX:-8192} PS5_CHUNK=${PS5_CHUNK:-512} PS5_KV=${PS5_KV:-q8} PS5_THREADS=${PS5_THREADS:-6}
args=(bench --model "$model" --pp "${PS5_PP:-32,512,2048,4096}" --tg "${PS5_TG:-128}" --reps "${PS5_REPS:-3}" --mtp 0 "$@")
"$here/diagnose.sh" > "$out/diagnostics.txt" 2>&1 || true
for case in base1 import1 import2 base2; do
  flag=0; [[ $case == import* ]] && flag=1
  rm -f "$out/$case.json"
  STRATA_PS5_ZERO_COPY=$flag "$here/run.sh" "${args[@]}" --json-out "$out/$case.json" 2>&1 | tee "$out/$case.log"
  test -s "$out/$case.json"
done
python3 "$here/summarize-r3.py" "$out"
