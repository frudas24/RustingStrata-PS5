# Sourced by build/test scripts; no changes to the system or the console.
PS5_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PS5_ROOT/target/ps5}"
unset CARGO_ENCODED_RUSTFLAGS
export RUSTFLAGS='-C target-cpu=znver2'
export CARGO_PROFILE_RELEASE_LTO="${PS5_LTO:-fat}"
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
export CARGO_PROFILE_RELEASE_DEBUG=0
PS5_TARGET=x86_64-unknown-linux-gnu
