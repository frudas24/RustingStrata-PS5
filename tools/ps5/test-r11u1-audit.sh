#!/usr/bin/env bash
# Run independent audit checks, then the actual Rust/Vulkan tests when explicitly requested.
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
export PYTHONDONTWRITEBYTECODE=1
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
case "${1:---offline}" in
  --offline)
    python3 tools/ps5/check-r11u1-audit.py
    for n in 9 10 11; do python3 "tools/ps5/check-r$n.py"; done
    python3 tools/test_kld_compare.py
    ;;
  --cpu)
    log=$(mktemp); trap 'rm -f "$log"' EXIT
    checked_one() {
      cargo test "${args[@]}" "$@" -- --exact --nocapture 2>&1 | tee "$log"
      grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored;' "$log"
    }
    checked_one --test gguf audit_rejects_inconsistent_tensor_layouts
    checked_one --test gguf audit_rejects_deep_metadata
    checked_one --bin rustingstrata audit_log_probs_rejects_invalid_input
    ;;
  --gpu)
    while IFS= read -r name; do unset "$name"; done < <(compgen -e | grep -E '^(STRATA_|PS5_)' || true)
    export STRATA_PS5=1 STRATA_VK_NO_COOP=1 STRATA_TEST_BACKENDS=vulkan
    log=$(mktemp); trap 'rm -f "$log"' EXIT
    cargo test "${args[@]}" --test ps5_audit moe_duplicate_slots_complete -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$log"
    grep -Fq 'AUDIT_MOE_DUPLICATES PASS: 5 shapes, including overflow' "$log"
    grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored;' "$log"
    ;;
  *) echo 'usage: test-r11u1-audit.sh [--offline|--cpu|--gpu]' >&2; exit 2;;
esac
