#!/usr/bin/env bash
# Run after test-r2.sh passes. Same model, settings, memory flags, and binary for every case.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
[[ $# -ge 1 ]] || { echo 'usage: bench-r2.sh MODEL.gguf [OUTPUT_DIR] [extra bench flags]' >&2; exit 2; }
model=$1; shift; [[ -r "$model" ]] || { echo 'model not readable' >&2; exit 2; }
out=${1:-ps5-r2-bench-$(date +%Y%m%d-%H%M%S)}; if (($#)); then shift; fi
mkdir -p "$out"
"$here/diagnose.sh" > "$out/diagnostics.txt" 2>&1 || true
export STRATA_PS5=1 STRATA_PS5_GTT=1 STRATA_PS5_DIRECT=1 STRATA_FIXED_EXPERTS=1 STRATA_VK_NO_COOP=1
export STRATA_PS5_ZERO_COPY=0
export PS5_CTX=${PS5_CTX:-8192} PS5_CHUNK=${PS5_CHUNK:-512} PS5_KV=${PS5_KV:-q8} PS5_THREADS=${PS5_THREADS:-6}
# Short prompts expose the small dense tile; long prompts measure normal prefill.
args=(bench --model "$model" --pp "${PS5_PP:-32,64,512,2048,4096}" --tg "${PS5_TG:-128}" --reps "${PS5_REPS:-3}" --mtp 0 "$@")
failed=0
for case in r1 r2-64 r2-128 r2-small r2-combined-64 r2-combined-128; do
  wg=256; small=0
  case "$case" in r2-64) wg=64;; r2-128) wg=128;; r2-small) small=1;; r2-combined-64) wg=64; small=1;; r2-combined-128) wg=128; small=1;; esac
  rm -f "$out/$case.json"
  if STRATA_PS5_GEMV_WG=$wg STRATA_PS5_SMALL_GEMM=$small "$here/run.sh" "${args[@]}" --json-out "$out/$case.json" 2>&1 | tee "$out/$case.log"; then
    [[ -s "$out/$case.json" ]] || failed=1
  else failed=1; fi
done
python3 "$here/summarize-r2.py" "$out" || failed=1
exit "$failed"
