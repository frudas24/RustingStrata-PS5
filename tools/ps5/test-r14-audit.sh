#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
export PYTHONDONTWRITEBYTECODE=1
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
case "${1:---offline}" in
  --offline)
    python3 tools/ps5/check-r14.py
    ;;
  --cpu)
    log=$(mktemp); trap 'rm -f "$log"' EXIT
    for spec in 'lib r14_dispatch_contract' 'bin r14_fixed_logit_writer'; do
      read -r kind name <<< "$spec"
      target=(--lib); [[ $kind != bin ]] || target=(--bin rustingstrata)
      cargo test "${args[@]}" "${target[@]}" "$name" -- --nocapture 2>&1 | tee "$log"
      python3 - "$log" <<'PY'
import importlib.util,sys
from pathlib import Path
p=Path('tools/ps5/r14-campaign.py');s=importlib.util.spec_from_file_location('r14',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.check_test_log(Path(sys.argv[1]).read_text())
PY
    done
    ;;
  --context|--kv-growth)
    [[ $# == 3 ]] || { echo 'usage: test-r14-audit.sh [--context|--kv-growth] MODEL_GGUF NEW_LOG' >&2; exit 2; }
    [[ -f $2 && ! -e $3 ]] || { echo 'model missing or log already exists' >&2; exit 1; }
    model=$(realpath "$2"); log=$(realpath -m "$3")
    while IFS= read -r name; do unset "$name"; done < <(compgen -A variable STRATA_)
    export STRATA_PS5=1 STRATA_VK_NO_COOP=1 STRATA_FIXED_EXPERTS=1 STRATA_KV=q8 STRATA_KV_K=q8 STRATA_KV_V=q8 STRATA_KV_Q8=1 STRATA_PS5_R11_IQ32=1 STRATA_PS5_R12_IQ_INT_TILE=64 STRATA_PS5_R13_LUT=off RAYON_NUM_THREADS=6
    if [[ $1 == --kv-growth ]]; then
      PS5_R14_DENSE_MODEL="$model" cargo test "${args[@]}" --lib gpu_model::tests::r14_kv_growth_preserves_slots -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$log"
      marker='R14 KV GROWTH PASS: old bytes, other slot, shrink and reuse'
    else
      PS5_R14_CONTEXT_MODEL="$model" cargo test "${args[@]}" --test r14_context greedy_context_boundary -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$log"
      marker='R14 GREEDY CONTEXT PASS: exact limit, inert rejection, reset and replay'
    fi
    [[ $(grep -Fxc "$marker" "$log" || true) == 1 ]]
    [[ $(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored;' "$log" || true) == 1 ]]
    [[ $(grep -c '^test result:' "$log" || true) == 1 ]]
    ;;
  *) echo 'usage: test-r14-audit.sh [--offline|--cpu|--context MODEL_GGUF NEW_LOG|--kv-growth DENSE_GGUF NEW_LOG]' >&2; exit 2;;
esac
