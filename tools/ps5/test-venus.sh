#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$here/../.."
case ${1:---cpu} in
  --cpu)
    # Host functional test: no PS5 flags or shader compiler required.
    export STRATA_PS5_VENUS_F16=0 STRATA_PS5_VENUS_MATVEC=0 STRATA_PS5_VENUS_BF16=0
    cargo test --locked --release --lib venus_ -- --nocapture
    if [[ $# -ge 2 ]]; then
      VENUS_F16_MODEL="$2" cargo test --locked --release --lib venus_f16_real_gguf -- --ignored --nocapture
    fi
    ;;
  --model|--bf16-model)
    [[ $# == 3 ]] || { echo 'usage: test-venus.sh --model|--bf16-model model.gguf result-directory' >&2; exit 2; }
    flag=STRATA_PS5_VENUS_MATVEC; marker=f16_matvec
    if [[ $1 == --bf16-model ]]; then flag=STRATA_PS5_VENUS_BF16; marker=bf16_matvec; fi
    export VENUS_F16_MODEL="$2" STRATA_PS5_VENUS_F16=0 STRATA_PS5_VENUS_MATVEC=0 STRATA_PS5_VENUS_BF16=0 STRATA_PS5_VENUS_LOG=1
    unset STRATA_SCALAR STRATA_PS5_VENUS_TRACE
    mkdir -p "$3"
    for mode in 0 1; do
      env "$flag=$mode" VENUS_CPU_OUT="$3/$mode.json" cargo test --locked --release --test venus_cpu venus_cpu_model -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$3/$mode.log"
    done
    grep -q "PS5 Venus exercised: $marker" "$3/1.log"
    if grep -q "PS5 Venus exercised: $marker" "$3/0.log"; then exit 1; fi
    python3 "$here/check-venus-cpu.py" "$3/0.json" "$3/1.json"
    ;;
  --trace)
    [[ $# == 4 ]] || { echo 'usage: test-venus.sh --trace model.gguf layer:position:rank output.log' >&2; exit 2; }
    export STRATA_MODEL="$2" STRATA_PS5_VENUS_TRACE="$3" STRATA_PS5_VENUS_TRACE_VALUES=1
    export STRATA_PS5=1 STRATA_PS5_SCHED=off STRATA_PS5_ZERO_COPY=0 STRATA_PS5_VENUS_F16=0 STRATA_PS5_VENUS_MATVEC=0 STRATA_PS5_VENUS_BF16=0
    export STRATA_PS5_R7_GEMM_TILE=0 STRATA_PS5_GEMV_WG=256 STRATA_PS5_SMALL_GEMM=0 STRATA_PS5_ATTN_WG=0
    unset STRATA_NAN_TRACE STRATA_LLAMA_NUMERICS STRATA_KV_HOST STRATA_VK_TURBO
    source "$here/env.sh"
    cargo test --locked --release --target "$PS5_TARGET" --features ps5-r2 --test venus_trace venus_trace_real_moe -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$4"
    python3 "$here/check-venus-trace.py" "$4" "$3"
    ;;
  *) echo 'usage: test-venus.sh --cpu [F16 GGUF] | --model F16.gguf results | --bf16-model BF16.gguf results | --trace MoE.gguf layer:position:rank output.log' >&2; exit 2;;
esac
