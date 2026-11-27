#!/usr/bin/env bash
# Physical gate, never count a filtered or skipped Rust test as a pass.
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
while IFS= read -r name; do unset "$name"; done < <(compgen -A variable STRATA_)
export STRATA_PS5=1 STRATA_VK_NO_COOP=1 STRATA_TEST_BACKENDS=vulkan
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
log=$(mktemp); trap 'rm -f "$log"' EXIT
run_case() {
    local name=$1 marker=$2
    cargo test "${args[@]}" --test r13_launch "$name" -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$log"
    [[ $(grep -Fxc "$marker" "$log" || true) == 1 ]]
    [[ $(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored;' "$log" || true) == 1 ]]
    [[ $(grep -c '^test result:' "$log" || true) == 1 ]]
}
run_case q8_graph_cache_lifetime 'R13 LAUNCH Q8 GRAPH PASS: replay, discard and scratch growth'
run_case rejected_graph_operations_are_inert 'R13 FINAL GRAPH PASS: rejected operations inert, device copy disjoint'
