# R6 continuity

R5 base SHA-256: `ef8b2c85539abdf0e665c2cdd90a4297feaffdbcfba203bd5cd9d5f349560261`

See PS5-R6.md and validation-r6.txt.

Modified inherited files:
- `README.md`
- `src/gpu_model/prefill.rs`
- `src/main.rs`
- `src/vulkan/mod.rs`
- `tools/ps5/env.sh`

New files (excluding this manifest):
- `docs/PS5-R6.md`
- `docs/validation-r6.txt`
- `tests/ps5_r6.rs`
- `tools/ps5/autotune-r6.py`
- `tools/ps5/check-r6.py`
- `tools/ps5/pgo-r6.sh`
- `tools/ps5/probe-r6.c`
- `tools/ps5/probe-r6.sh`
- `tools/ps5/test-r6.sh`

No inherited files removed. No real hardware speedup claimed. R7 should use real PS5 measurements before new speculative kernels.
