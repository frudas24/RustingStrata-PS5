#!/usr/bin/env bash
# Four-arm single-row replay; all four use GPU-only scheduling. Never benchmark.
set -euo pipefail
[[ $# == 2 && -r $1 ]] || { echo 'usage: test-r9.sh GENERIC-MOE.gguf NEW_OUTPUT_DIR' >&2; exit 2; }
model=$(realpath "$1"); out=$(realpath -m "$2"); mkdir "$out"
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
# Clear inherited experimental flags before building and running.
while IFS= read -r name; do unset "$name"; done < <(compgen -A variable STRATA_)
export STRATA_MODEL="$model" STRATA_PS5=1 STRATA_PS5_GTT=1 STRATA_PS5_DIRECT=1 STRATA_TEST_BACKENDS=vulkan
export STRATA_PS5_GEMV_WG=256 STRATA_PS5_SMALL_GEMM=0 STRATA_PS5_ATTN_WG=0 STRATA_PS5_ZERO_COPY=0 STRATA_PS5_R7_GEMM_TILE=0 STRATA_VK_NO_COOP=1
export STRATA_PS5_R4_RESEARCH=1 STRATA_PS5_SCHED_SCRATCH_MB=64 STRATA_PS5_SCHED_MAX_ROWS=32 STRATA_FIXED_EXPERTS=1 STRATA_EXPERT_MB=0 RAYON_NUM_THREADS=6
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
git rev-parse HEAD > "$out/commit.txt"
cargo test "${args[@]}" --lib ps5_r8::tests 2>&1 | tee "$out/unit.log"
cargo test "${args[@]}" --test ps5_r4 --no-run --message-format=json-render-diagnostics > "$out/build.jsonl"
bin=$(python3 - "$out/build.jsonl" <<'PY'
import json,sys
paths=[r['executable'] for s in open(sys.argv[1]) if (r:=json.loads(s)).get('target',{}).get('name')=='ps5_r4' and r.get('executable')]
assert len(paths)==1,paths
print(paths[0])
PY
)
sha256sum "$bin" > "$out/binary.sha256"
rc=0
for arm in gpu shadow legacy contract; do
  replay=; probe=0; if [[ $arm != gpu ]]; then replay=$arm; probe=1; fi
  # Environment is per process, never shared or changed by the Rust executor.
  STRATA_PS5_SCHED=gpu STRATA_R7_R4_STAGE_PROBE="$probe" STRATA_PS5_R9_MODE="$replay" \
    STRATA_PS5_R9_LAYER="${PS5_R9_LAYER:-0}" STRATA_PS5_R9_POS="${PS5_R9_POS:-0}" STRATA_PS5_R9_TOKEN="${PS5_R9_TOKEN:-0}" STRATA_PS5_R9_RANK="${PS5_R9_RANK:-0}" \
    PS5_R4_PARITY_FILE="$out/$arm.json" \
    python3 tools/ps5/r8-run-guarded.py --out "$out/$arm" --timeout "${PS5_R9_TIMEOUT:-1800}" --min-kib 1048576 -- \
    "$bin" prefill_scheduler_parity --ignored --exact --nocapture --test-threads=1 > "$out/$arm.log" 2>&1 || { echo "$arm failed, see log" >&2; rc=1; }
done
((rc==0)) || exit 1
python3 tools/ps5/compare-r9.py "$out"
