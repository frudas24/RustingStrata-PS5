#!/usr/bin/env bash
# Full routed and independent dense oracle, separate processes for each candidate.
set -euo pipefail
[[ $# == 1 ]] || { echo 'usage: test-r13-lut.sh NEW_OUTPUT_DIR' >&2; exit 2; }
out=$(realpath -m "$1"); mkdir "$out"
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
while IFS= read -r name; do unset "$name"; done < <(compgen -A variable STRATA_)
export STRATA_PS5=1 STRATA_VK_NO_COOP=1 STRATA_PS5_R6=1 STRATA_TEST_BACKENDS=vulkan
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
git rev-parse HEAD > "$out/commit.txt"
cargo test "${args[@]}" --test r13_lut lds_budget -- --exact --nocapture 2>&1 | tee "$out/lds.log"
[[ $(grep -c '^R13 LDS ' "$out/lds.log") == 7 ]]
# Check exact set of dispatch markers: absence is not a successful fallback.
check_marks() {
  python3 tools/ps5/r13-campaign.py marks --log "$1" --test "$2" --tile "$3" --lut "$4"
}
for lut in off iq2; do
  STRATA_PS5_R11_IQ32=1 STRATA_PS5_R13_LUT=$lut PS5_R11_GPU_FILE="$out/iq2-$lut.bin" cargo test "${args[@]}" --test r11_iq fingerprint -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$out/iq2-$lut.log"
  [[ $(stat -c %s "$out/iq2-$lut.bin") == 17905536 ]]
  grep -Fq 'R11 IQ GPU parity: 648 cases' "$out/iq2-$lut.log"
  check_marks "$out/iq2-$lut.log" iq2 0 "$lut"
  cmp "$out/iq2-off.bin" "$out/iq2-$lut.bin"
done
for tile in 0 32 64; do
  for lut in off int; do
    STRATA_PS5_R12_IQ_INT_TILE=$tile STRATA_PS5_R13_LUT=$lut PS5_R12_GPU_FILE="$out/int-$tile-$lut.bin" cargo test "${args[@]}" --test r12_iq fingerprint -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$out/int-$tile-$lut.log"
    [[ $(stat -c %s "$out/int-$tile-$lut.bin") == 37981440 ]]
    grep -Fq 'R12 IQ integer GPU parity: 1296 cases' "$out/int-$tile-$lut.log"
    check_marks "$out/int-$tile-$lut.log" int "$tile" "$lut"
    cmp "$out/int-0-off.bin" "$out/int-$tile-$lut.bin"
  done
done
sha256sum "$out"/*.bin > "$out/gpu.sha256"
echo 'PASS: R13 LUT isolated full-buffer parity, including all three integer geometries.'
