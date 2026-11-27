#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
export PYTHONDONTWRITEBYTECODE=1
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
case "${1:---offline}" in
  --offline)
    python3 tools/ps5/check-r13-audit.py
    python3 tools/ps5/check-r13.py
    bash tools/ps5/test-r12-audit.sh --offline
    python3 tools/ps5/check-r6.py
    ;;
  --cpu)
    log=$(mktemp); trap 'rm -f "$log"' EXIT
    cargo test "${args[@]}" --lib r13_audit_ -- --nocapture 2>&1 | tee "$log"
    grep -Eq '^test result: ok\. 2 passed; 0 failed; 0 ignored;' "$log"
    bash tools/ps5/test-r12-audit.sh --cpu
    ;;
  --gpu)
    [[ $# == 2 ]] || { echo 'usage: test-r13-audit.sh --gpu NEW_OUTPUT_DIR' >&2; exit 2; }
    bash tools/ps5/test-r11u1-audit.sh --gpu
    bash tools/ps5/test-r13-lut.sh "$2"
    ;;
  *) echo 'usage: test-r13-audit.sh [--offline|--cpu|--gpu NEW_OUTPUT_DIR]' >&2; exit 2;;
esac
