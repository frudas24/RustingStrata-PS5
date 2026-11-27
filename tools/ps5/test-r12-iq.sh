#!/usr/bin/env bash
# No model required: the actual Vulkan integer tile kernels use checked-in IQ fixtures.
set -euo pipefail
[[ $# == 1 ]] || { echo 'usage: test-r12-iq.sh NEW_OUTPUT_DIR' >&2; exit 2; }
out=$(realpath -m "$1"); mkdir "$out"
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
while IFS= read -r name; do unset "$name"; done < <(compgen -A variable STRATA_)
export STRATA_PS5=1 STRATA_VK_NO_COOP=1 STRATA_PS5_R6=1 STRATA_TEST_BACKENDS=vulkan
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
git rev-parse HEAD > "$out/commit.txt"
cargo test "${args[@]}" --test r12_iq shared_bound -- --exact --nocapture 2>&1 | tee "$out/lds.log"
grep -Fq 'R12 LDS type=18' "$out/lds.log"; grep -Fq 'R12 LDS type=23' "$out/lds.log"
for flag in 0 32 64; do
  STRATA_PS5_R12_IQ_INT_TILE=$flag PS5_R12_GPU_FILE="$out/gpu-$flag.bin" cargo test "${args[@]}" --test r12_iq fingerprint -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$out/gpu-$flag.log"
  [[ $(stat -c %s "$out/gpu-$flag.bin") == 37981440 ]]
  grep -Fq 'R12 IQ integer GPU parity: 1296 cases' "$out/gpu-$flag.log"
  for tile in 32 64; do
    if [[ $flag == "$tile" ]]; then grep -Fqx "PS5 R6 exercised: gemmi_dp_ps5_iq$tile" "$out/gpu-$flag.log"
    elif grep -Fqx "PS5 R6 exercised: gemmi_dp_ps5_iq$tile" "$out/gpu-$flag.log"; then
      echo "unexpected IQ tile $tile dispatched in arm $flag" >&2; exit 1
    fi
  done
  cmp "$out/gpu-0.bin" "$out/gpu-$flag.bin"
done
sha256sum "$out"/gpu-*.bin > "$out/gpu.sha256"
echo 'PASS: R12 routed integer GPU full-buffer identity, separate processes.'
