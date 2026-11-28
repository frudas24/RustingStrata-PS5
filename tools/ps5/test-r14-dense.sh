#!/usr/bin/env bash
# Existing dense shaders, new type-specific dispatch. Real Vulkan, no GGUF required.
set -euo pipefail
[[ $# == 1 ]] || { echo 'usage: test-r14-dense.sh NEW_OUTPUT_DIR' >&2; exit 2; }
out=$(realpath -m "$1"); mkdir "$out"
source "$(dirname -- "$0")/env.sh"
cd "$PS5_ROOT"
while IFS= read -r name; do unset "$name"; done < <(compgen -A variable STRATA_)
export STRATA_PS5=1 STRATA_VK_NO_COOP=1 STRATA_PS5_R6=1 STRATA_TEST_BACKENDS=vulkan
args=(--locked --release --target "$PS5_TARGET" --features ps5-r2)
git rev-parse HEAD > "$out/commit.txt"
cargo test "${args[@]}" --test r14_dense shared_bound -- --exact --nocapture 2>&1 | tee "$out/lds.log"
grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored;' "$out/lds.log"
for ty in 13 14; do grep -Fq "R14 LDS type=$ty " "$out/lds.log"; done
for tile in 0 32 64; do
  for types in both q5 q6; do
    if [[ $tile == 0 && $types != both ]]; then continue; fi
    stem="$out/$tile-$types"
    STRATA_PS5_R14_DENSE_TILE=$tile STRATA_PS5_R14_DENSE_TYPE=$types PS5_R14_GPU_FILE="$stem.bin" cargo test "${args[@]}" --test r14_dense fingerprint -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$stem.log"
    grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored;' "$stem.log"
    grep -Fqx 'R14 dense GPU parity: 498 cases 4486580 values' "$stem.log"
    [[ $(stat -c %s "$stem.bin") == 17946320 ]]
    python3 - "$stem.log" "$tile" "$types" <<'PY'
import importlib.util,sys
from pathlib import Path
p=Path('tools/ps5/r14-campaign.py');s=importlib.util.spec_from_file_location('r14',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.check_marks(Path(sys.argv[1]).read_text(),int(sys.argv[2]),sys.argv[3])
PY
    cmp "$out/0-both.bin" "$stem.bin"
  done
done
sha256sum "$out"/*.bin > "$out/gpu.sha256"
echo 'PASS: R14 dense full-buffer identity across seven isolated arms.'
