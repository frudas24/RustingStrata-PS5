# Sourced by build/test scripts; no changes to the system or the console.
PS5_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PS5_ROOT/target/ps5}"
unset CARGO_ENCODED_RUSTFLAGS
export RUSTFLAGS='-C target-cpu=znver2'
# R6: isolate PGO builds; preserve path boundaries with Cargo's encoded flags.
if [[ -n ${PS5_PGO_MODE:-} ]]; then
  [[ ${PS5_PGO_PATH:-} = /* ]] || { echo 'PGO path must be absolute' >&2; return 1; }
  case "$PS5_PGO_MODE" in
    generate) pgo="profile-generate=$PS5_PGO_PATH" ;;
    use) [[ -s $PS5_PGO_PATH ]] || { echo 'PGO merged profile missing' >&2; return 1; }; pgo="profile-use=$PS5_PGO_PATH" ;;
    *) echo 'PGO mode must be generate or use' >&2; return 1 ;;
  esac
  unset RUSTFLAGS
  export CARGO_ENCODED_RUSTFLAGS="-C"$'\x1f'"target-cpu=znver2"$'\x1f'"-C"$'\x1f'"$pgo"
  unset pgo
fi
export CARGO_PROFILE_RELEASE_LTO="${PS5_LTO:-fat}"
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
export CARGO_PROFILE_RELEASE_DEBUG=0
PS5_TARGET=x86_64-unknown-linux-gnu
