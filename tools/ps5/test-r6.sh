#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
mode=${1:---cpu}
[[ $mode == --cpu || $mode == --gpu ]] || { echo 'usage: test-r6.sh [--cpu|--gpu]' >&2; exit 2; }
python3 "$here/check-r6.py"
for r in 2 3 4 5; do python3 "$here/check-r$r.py"; done
"$here/probe-r6.sh" "$mode"
# Independent R1 primitive check, without requiring Rust for --cpu.
tmp=$(mktemp -d); trap 'rm -rf -- "$tmp"' EXIT
cc -O3 -march=znver2 -Wall -Wextra -Werror "$here/simd_check.c" -o "$tmp/simd"
"$tmp/simd"
