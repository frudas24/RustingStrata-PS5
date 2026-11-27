#!/usr/bin/env bash
# Build/unit checks or explicit physical batch quarantine test. No performance claims.
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
case "${1:---compile}" in
  --compile)
    bash tools/ps5/build.sh
    log=$(mktemp); trap 'rm -f "$log"' EXIT
    cargo test "${args[@]}" --lib u1_batch_tests -- --nocapture 2>&1 | tee "$log"
    grep -Eq '^test result: ok\. 3 passed; 0 failed; 0 ignored;' "$log"
    cargo test "${args[@]}" --lib layer_lookup_outside_partition_is_none -- --nocapture 2>&1 | tee "$log"
    grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored;' "$log"
    cargo test "${args[@]}" --test gpu_model --no-run
    for n in 9 10 11; do python3 "tools/ps5/check-r$n.py"; done
    python3 tools/test_kld_compare.py
    ;;
  --batch)
    if [[ $# != 3 || ! -f $2 ]]; then echo 'usage: test-r11u1.sh --batch /absolute/model.gguf /absolute/new-output.log' >&2; exit 2; fi
    model=$(realpath "$2")
    log=$(realpath -m "$3")
    [[ ! -e $log ]] || { echo "refusing to overwrite $log" >&2; exit 2; }
    # Eliminate ambient Strata/PS5 research flags from this fixed control.
    while IFS= read -r name; do unset "$name"; done < <(compgen -e | grep -E '^(STRATA_|PS5_)' || true)
    export STRATA_MODEL="$model" STRATA_PS5=1 STRATA_PS5_GTT=1 STRATA_PS5_DIRECT=1 STRATA_PS5_HOST_RESERVE_MB=1536
    export STRATA_EXPERT_MB=0 STRATA_FIXED_EXPERTS=1 STRATA_PS5_SCHED=off STRATA_PS5_U1_BATCH_LOG=1
    export STRATA_VK_NO_COOP=1 STRATA_TEST_BACKENDS=vulkan RAYON_NUM_THREADS=6
    mkdir -p "$(dirname "$log")"
    for frac in 0 1; do
      [[ ! -e "$log.$frac.bits" && ! -e "$log.$frac.log" ]] || { echo 'arm output already exists' >&2; exit 2; }
      STRATA_CPU_FRAC=$frac STRATA_PS5_U1_BATCH_OUT="$log.$frac.bits" \
        cargo test "${args[@]}" --test gpu_model u1_ps5_batch_quarantine -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$log.$frac.log"
    done
    python3 tools/ps5/verify-r11u1-batch.py "$log"
    printf '%s\n' 'PASS: both arms exercised, full logits identical' > "$log"
    ;;
  *) echo 'usage: test-r11u1.sh [--compile | --batch model.gguf new-output.log]' >&2; exit 2;;
esac
