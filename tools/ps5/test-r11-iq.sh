#!/usr/bin/env bash
# CPU contract and actual Vulkan routed IQ2 parity. No production profile changes.
set -euo pipefail
[[ $# == 2 && -r $1 ]] || { echo 'usage: test-r11-iq.sh MOE.gguf NEW_OUTPUT_DIR' >&2; exit 2; }
model=$(realpath "$1"); out=$(realpath -m "$2"); mkdir "$out"
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
while IFS= read -r name; do unset "$name"; done < <(compgen -A variable STRATA_)
export STRATA_PS5=1 STRATA_PS5_GTT=1 STRATA_PS5_DIRECT=1 STRATA_VK_NO_COOP=1 STRATA_PS5_R6=1 STRATA_PS5_SCHED=off RAYON_NUM_THREADS=6
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
git rev-parse HEAD > "$out/commit.txt"
sha256sum "$model" > "$out/model.sha256"
cargo test "${args[@]}" --lib ps5_r11::tests -- --nocapture 2>&1 | tee "$out/cpu-unit.log"
PS5_R11_MODEL="$model" cargo test "${args[@]}" --lib ps5_r11::tests::real_iq_bench -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$out/cpu-real.log"
if [[ -n ${PS5_R11_SHADOW:-} ]]; then
  PS5_R11_MODEL="$model" cargo test "${args[@]}" --lib ps5_r11::tests::captured_iq_bench -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$out/cpu-captured.log"
  [[ $(grep -c '^R11_IQ_BENCH ' "$out/cpu-captured.log") == 3 ]]
fi
[[ $(grep -c '^R11_IQ_BENCH ' "$out/cpu-real.log") == 3 ]]
cargo test "${args[@]}" --test r11_iq shared_bound -- --nocapture 2>&1 | tee "$out/lds.log"
for flag in 0 1; do
  STRATA_PS5_R11_IQ32=$flag PS5_R11_GPU_FILE="$out/gpu-$flag.bin" cargo test "${args[@]}" --test r11_iq fingerprint -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$out/gpu-$flag.log"
  test -s "$out/gpu-$flag.bin"
  grep -Fq 'R11 IQ GPU parity: 648 cases' "$out/gpu-$flag.log"
done
! grep -Fq 'PS5 R6 exercised: gemm_f_dp_ps5_iq32' "$out/gpu-0.log"
grep -Fq 'PS5 R6 exercised: gemm_f_dp_ps5_iq32' "$out/gpu-1.log"
cmp "$out/gpu-0.bin" "$out/gpu-1.bin"
sha256sum "$out"/gpu-*.bin > "$out/gpu.sha256"
echo 'PASS: IQ2 routed GPU parity. CPU benchmark compares separate contracts; no production promotion.'
