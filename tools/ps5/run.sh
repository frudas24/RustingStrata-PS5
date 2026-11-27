#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/../.." && pwd)
if (( $# < 1 )); then echo 'usage: run.sh {chat|generate|bench|serve|perplexity|score} [args...]' >&2; exit 2; fi
sub=$1; shift
case "$sub" in chat|generate|bench|serve|perplexity|score) ;; *) echo "Unsupported subcommand for PS5 preset: $sub" >&2; exit 2;; esac
bin=${STRATA_PS5_BIN:-${CARGO_TARGET_DIR:-$root/target/ps5}/x86_64-unknown-linux-gnu/release/rustingstrata}
[[ -x "$bin" ]] || { echo "Build first: $root/tools/ps5/build.sh" >&2; exit 1; }
export STRATA_PS5=${STRATA_PS5:-1}
export STRATA_PS5_SCHED=${STRATA_PS5_SCHED:-off}
export STRATA_PS5_GTT=${STRATA_PS5_GTT:-1}
export STRATA_PS5_DIRECT=${STRATA_PS5_DIRECT:-1}
export STRATA_PS5_HOST_RESERVE_MB=${STRATA_PS5_HOST_RESERVE_MB:-1536}
export STRATA_VK_ATTN_BUDGET=${STRATA_VK_ATTN_BUDGET:-16777216}
args=("$@")
has() { local a; for a in "${args[@]}"; do [[ $a == "$1" || $a == "$1="* ]] && return 0; done; return 1; }
value() { local i; for ((i=0;i<${#args[@]};i++)); do
  if [[ ${args[i]} == "$1="* ]]; then printf '%s' "${args[i]#*=}"; return; fi
  if [[ ${args[i]} == "$1" ]]; then printf '%s' "${args[i+1]:-}"; return; fi
 done; printf '%s' "$2"; }
defaults=()
add() { if ! has "$1"; then defaults+=("$@"); fi; }
add --gpu
add --backend vulkan
add --device 0
add --ctx "${PS5_CTX:-4096}"
add --chunk "${PS5_CHUNK:-512}"
add --kv "${PS5_KV:-q8}"
add --threads "${PS5_THREADS:-6}"
add --vram-margin "${PS5_VRAM_MARGIN:-768}"
add --ram-margin "$STRATA_PS5_HOST_RESERVE_MB"
add --cache-ram "${PS5_CACHE_RAM:-256}"
chunk=$(value --chunk "${PS5_CHUNK:-512}")
export STRATA_DS_CHUNK=${STRATA_DS_CHUNK:-$chunk}
export STRATA_GLM_CHUNK=${STRATA_GLM_CHUNK:-$chunk}
export STRATA_G4_CHUNK=${STRATA_G4_CHUNK:-$chunk}
exec "$bin" "$sub" "${defaults[@]}" "${args[@]}"
