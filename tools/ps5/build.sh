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
features=()
if [[ ${PS5_R2_BUILD:-1} == 1 ]]; then
  command -v "${GLSLC:-glslc}" >/dev/null || { echo "Install glslc or set PS5_R2_BUILD=0 for R1 shaders." >&2; exit 1; }
  command -v "${SPIRV_VAL:-spirv-val}" >/dev/null || { echo "Install spirv-tools or set PS5_R2_BUILD=0." >&2; exit 1; }
  features=(--features ps5-r2)
fi
cargo build "${features[@]}" --locked --release --target "$PS5_TARGET" --bin rustingstrata "$@"
printf 'PS5 binary: %s/%s/release/rustingstrata\n' "$CARGO_TARGET_DIR" "$PS5_TARGET"
