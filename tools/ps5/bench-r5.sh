#!/usr/bin/env bash
# Run after test-r5.sh. No fallback accepted in candidate arms.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
[[ $# -ge 1 && -r $1 ]] || { echo 'usage: bench-r5.sh MODEL.gguf [OUTPUT_DIR]' >&2; exit 2; }
model=$(realpath "$1"); out=${2:-ps5-r5-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$out"
export STRATA_PS5=1 STRATA_PS5_GTT=1 STRATA_PS5_DIRECT=1 STRATA_PS5_ZERO_COPY=0 STRATA_PS5_SCHED=off STRATA_FIXED_EXPERTS=1 STRATA_VK_NO_COOP=1
export STRATA_PS5_GEMV_WG=256 STRATA_PS5_SMALL_GEMM=0
unset STRATA_VK_DP_OLD STRATA_LLAMA_NUMERICS
export PS5_CTX=${PS5_CTX:-65536} PS5_CHUNK=${PS5_CHUNK:-256} PS5_KV=${PS5_KV:-q8} PS5_THREADS=${PS5_THREADS:-6}
export STRATA_VK_ATTN_BUDGET=${STRATA_VK_ATTN_BUDGET:-16777216}
pp=${PS5_PP:-1024,8192,32768,65024}; tg=${PS5_TG:-32}; reps=${PS5_REPS:-2}
python3 - "$pp" "$tg" "$PS5_CTX" <<'PY'
import sys
p=list(map(int,sys.argv[1].split(','))); tg,ctx=map(int,sys.argv[2:]); assert len(p)==len(set(p)) and min(p)>0 and tg>0 and max(p)+tg<=ctx, 'prompt + decode exceed context'
PY
"$here/diagnose.sh" > "$out/diagnostics.txt" 2>&1 || true
printf '%s\n' "ctx=$PS5_CTX" "kv=$PS5_KV" "chunk=$PS5_CHUNK" "budget=$STRATA_VK_ATTN_BUDGET" "pp=$pp" "tg=$tg" "reps=$reps" > "$out/config.txt"
for arm in 0a 256a 512a 1024a 1024b 512b 256b 0b; do
  wg=${arm%?}; rm -f "$out/$arm.json"
  STRATA_PS5_ATTN_WG=$wg "$here/run.sh" bench --model "$model" --pp "$pp" --tg "$tg" --reps "$reps" --mtp 0 --json-out "$out/$arm.json" 2>&1 | tee "$out/$arm.log"
  test -s "$out/$arm.json"
  if ((wg)); then grep -Fq "PS5 R5 attention dispatch: attn_prefill_ps5_$wg" "$out/$arm.log"; fi
done
python3 "$here/summarize-r5.py" "$out" "$pp"
