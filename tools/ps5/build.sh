#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
command -v cargo >/dev/null || { echo 'Install stable Rust (cargo + rustc) first.' >&2; exit 1; }
command -v rustc >/dev/null
rustc --version
rustc --print cfg --target "$PS5_TARGET" -C target-cpu=znver2 | grep -E 'target_feature="(avx2|fma|f16c)"'
for feat in avx2 fma f16c; do
  rustc --print cfg --target "$PS5_TARGET" -C target-cpu=znver2 | grep -Fxq "target_feature=\"$feat\"" || exit 1
done
cargo build --locked --release --target "$PS5_TARGET" --bin rustingstrata "$@"
printf 'PS5 binary: %s/%s/release/rustingstrata\n' "$CARGO_TARGET_DIR" "$PS5_TARGET"
