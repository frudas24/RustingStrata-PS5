#!/usr/bin/env bash
# Build separate instrumented + optimized binaries; never overwrite the normal R6 binary.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
[[ $# == 3 ]] || { echo 'usage: pgo-r6.sh MODEL.gguf APPROVED_PROFILE.json NEW_OUTPUT_DIR' >&2; exit 2; }
model=$(realpath "$1"); profile=$(realpath "$2"); out=$(realpath -m "$3")
[[ ! -e $out ]] || { echo 'Use a new output directory' >&2; exit 2; }
command -v rustc >/dev/null; command -v cargo >/dev/null
host=$(rustc -vV | sed -n 's/^host: //p')
profdata="$(rustc --print sysroot)/lib/rustlib/$host/bin/llvm-profdata"
[[ -x $profdata ]] || { echo 'Install llvm-tools-preview for this Rust toolchain (rustup component add llvm-tools-preview).' >&2; exit 1; }
base=${STRATA_PS5_BIN:-$root/target/ps5/x86_64-unknown-linux-gnu/release/rustingstrata}
base=$(realpath "$base"); [[ -x $base ]] || { echo 'Build the normal R6 binary first' >&2; exit 1; }
mkdir -p "$out/raw"
rustc -vV > "$out/rustc.txt"; "$profdata" --version > "$out/llvm-profdata.txt"
# Rustc's matching LLVM tool avoids raw-format mismatches. Explicit target keeps host build scripts uninstrumented.
CARGO_TARGET_DIR="$out/instrumented" PS5_PGO_MODE=generate PS5_PGO_PATH="$out/raw" bash "$here/build.sh"
inst="$out/instrumented/x86_64-unknown-linux-gnu/release/rustingstrata"
python3 "$here/autotune-r6.py" train-pgo --model "$model" --profile "$profile" --bin "$base" --instrumented-bin "$inst" --raw-dir "$out/raw" --out "$out/training"
shopt -s nullglob
raw=("$out/raw/"*.profraw); ((${#raw[@]} > 0)) || { echo 'No PGO data' >&2; exit 1; }
"$profdata" merge -o "$out/merged.profdata" "${raw[@]}"
CARGO_TARGET_DIR="$out/optimized" PS5_PGO_MODE=use PS5_PGO_PATH="$out/merged.profdata" bash "$here/build.sh"
optimized="$out/optimized/x86_64-unknown-linux-gnu/release/rustingstrata"
python3 "$here/autotune-r6.py" compare-pgo --model "$model" --profile "$profile" --bin "$base" --candidate-bin "$optimized" --out "$out/comparison"
printf 'PGO candidate passed. Retune with --bin %q to get its own profile.\n' "$optimized"
