#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
export PYTHONDONTWRITEBYTECODE=1
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
case "${1:---offline}" in
  --offline)
    python3 tools/ps5/check-r12-audit.py
    python3 tools/ps5/check-r12.py
    bash tools/ps5/test-r11u1-audit.sh --offline
    ;;
  --cpu)
    log=$(mktemp); trap 'rm -f "$log"' EXIT
    cargo test "${args[@]}" --lib r12_audit_ -- --nocapture 2>&1 | tee "$log"
    grep -Eq 'test result: ok\. 2 passed;' "$log"
    bash tools/ps5/test-r11u1-audit.sh --cpu
    ;;
  --gpu)
    [[ $# == 2 ]] || { echo 'usage: test-r12-audit.sh --gpu NEW_OUTPUT_DIR' >&2; exit 2; }
    bash tools/ps5/test-r11u1-audit.sh --gpu
    bash tools/ps5/test-r12-iq.sh "$2"
    ;;
  *) echo 'usage: test-r12-audit.sh [--offline|--cpu|--gpu NEW_OUTPUT_DIR]' >&2; exit 2;;
esac
