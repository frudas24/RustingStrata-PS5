# Resultados PS5 R11

| Prueba | Estado |
|---|---|
| build | 0 |
| check-r9 | 0 |
| check-r10 | 0 |
| check-r11 | 0 |
| r7 | 0 |
| regression-completed | 0 |
| causal-q8-config | 0 |
| r10-base | 0 |
| r11-contract | 0 |
| r11-iq | 0 |
| r10-base-layer1 | 0 |
| r10-base-rank3 | 0 |
| r11-contract-layer1 | 0 |
| r11-contract-rank3 | 0 |
| causal-completed | 0 |
| venus-tail-unit | 0 |
| venus-tail-gguf | 0 |
| iq32-8k | 0 |
| iq32-profile | 0 |
| iq32-quality | 0 |
| iq32-quality-extended | 0 |
| iq32-32k-full-grid | PARTIAL_REPLACED_BY_TARGETED_CONTEXT_RUN |
| rpw-screen-powerloss | INTERRUPTED |
| rpw-screen | 1 |
| rpw-screen-rechecked | 0 |
| iq32-timed8k | 0 |
| iq32-target-32k-powerloss | INTERRUPTED |
| iq32-target-32k | 0 |
| iq32-target-64k | 0 |
| combined | NOT_RUN_NO_COMPATIBLE_FINALISTS |
| campaign-completed | 0 |
| combined-completed | 0 |

## Comparación causal

| Campaña | Prompt | Prefill natural | Prefill congelado | Decode natural | Decode tras prefill congelado |
|---|---|---:|---:|---:|---:|
| r10-base | prompt-0 | 13.7031 | 0.916263 | 1.84523 | 0.951664 |
| r10-base | prompt-1 | 1.46677 | 0.317235 | 0.617406 | 0.671361 |
| r10-base-layer1 | prompt-0 | 7.92315 | 0.546266 | 1.82995 | 1.44204 |
| r10-base-layer1 | prompt-1 | 1.07533 | 0.572111 | 0.59954 | 0.643796 |
| r10-base-rank3 | prompt-0 | 4.98686 | 1.53965 | 1.44974 | 1.2736 |
| r10-base-rank3 | prompt-1 | 1.11653 | 0.292592 | 0.525583 | 0.721312 |
| r11-contract | prompt-0 | 7.87667 | 2.65332 | 1.88418 | 1.68381 |
| r11-contract | prompt-1 | 1.62135 | 0.272937 | 0.815643 | 0.536518 |
| r11-contract-layer1 | prompt-0 | 5.26457 | 1.35073 | 1.28695 | 1.33951 |
| r11-contract-layer1 | prompt-1 | 1.26111 | 0.47295 | 0.880655 | 0.761286 |
| r11-contract-rank3 | prompt-0 | 13.9885 | 1.69829 | 1.89748 | 1.413 |
| r11-contract-rank3 | prompt-1 | 1.11653 | 0.292592 | 0.525583 | 0.721312 |

Valores de error máximo absoluto de logits. El replay congela conjuntamente IDs y pesos durante prefill.

## IQ CPU

| Captura | Tensor | Formato | Referencia µs | Candidato µs | Legado Q8K µs |
|---|---|---|---:|---:|---:|
| matched-r9-shadow | ffn_gate_exps.weight | IQ2_XS | 881.709 | 737.402 | 37.362 |
| matched-r9-shadow | ffn_up_exps.weight | IQ2_XS | 883.777 | 752.352 | 37.398 |
| matched-r9-shadow | ffn_down_exps.weight | IQ3_XXS | 404.692 | 399.780 | 32.862 |
| deterministic | ffn_gate_exps.weight | IQ2_XS | 717.059 | 577.976 | 38.672 |
| deterministic | ffn_down_exps.weight | IQ3_XXS | 402.738 | 400.536 | 38.516 |
| deterministic | ffn_down_exps.weight | IQ4_XS | 258.530 | 259.837 | 24.443 |

## Rendimiento y calidad GPU

El barrido RPW usa el resultado de la recomputación sobre los brazos físicos. El JSON original y su error de agregación se conservan en su carpeta.

| Ejecución | Veredicto | Detalle |
|---|---|---|
| r11-iq32-32k | False | child exited -15; see /var/tmp/ps5-r11-validation.xVfKJP/source/results/r11-iq32-32k/prose-1-tile0.log |
| r11-iq32-8k | True | holdout ×1.2188, prose ×1.1983 |
| r11-iq32-near32k/iq32 | Filtro True | índice ×1.0960, peor ×1.0206 |
| r11-iq32-near64k/iq32 | Filtro True | índice ×1.0542, peor ×0.9985 |
| r11-iq32-near8k/iq32 | Filtro True | índice ×1.1582, peor ×1.0338 |
| r11-iq32-profile | False |  |
| r11-iq32-quality | True |  |
| r11-iq32-quality-extended | True |  |
| r11-rpw-screen/down1 | Filtro False | índice ×0.9622, peor ×0.9106 |
| r11-rpw-screen/down2 | Filtro False | índice ×0.9837, peor ×0.9444 |
| r11-rpw-screen/down4 | Filtro False | candidate dispatch absent or leaked |
| r11-rpw-screen/down8 | Filtro False | índice ×1.0119, peor ×0.9999 |
| r11-rpw-screen/gemv1 | Filtro False | índice ×1.0070, peor ×1.0009 |
| r11-rpw-screen/gemv2 | Filtro False | candidate dispatch absent or leaked |
| r11-rpw-screen/gemv4 | Filtro False | índice ×1.0009, peor ×0.9909 |
| r11-rpw-screen/gemv8 | Filtro False | índice ×0.9964, peor ×0.9726 |

## Venus Tail

22/22 geometrías.

## Contexto cercano al límite

| Brazo | Contexto / PP | Prefill t/s | Decode t/s | TTFT mediana ms | Mediana / p95 entre tokens ms | RSS máx. MiB | VRAM / GTT máx. MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| results/r11-iq32-near32k/baseline-first | 32768 / 32256 | 206.20 | 82.40 | 156471.88 | 11.68 / 14.29 | 1939.00 | 504.45 / 11641.98 |
| results/r11-iq32-near32k/iq32 | 32768 / 32256 | 242.70 | 84.10 | 132929.55 | 11.84 / 12.04 | 1883.73 | 504.44 / 11637.99 |
| results/r11-iq32-near32k-powerloss/baseline-first | 32768 / 32256 | 206.00 | 83.00 | 156598.43 | 11.77 / 13.56 | 1796.07 | 505.26 / 11629.98 |
| results/r11-iq32-near64k/baseline-first | 65536 / 65024 | 139.80 | 67.40 | 465293.58 | 14.75 / 14.90 | 1783.38 | 505.66 / 11977.98 |
| results/r11-iq32-near64k/iq32 | 65536 / 65024 | 155.60 | 67.30 | 418039.90 | 14.76 / 14.90 | 1883.92 | 507.41 / 11977.98 |
| results/r11-iq32-near8k/baseline-first | 8192 / 8128 | 314.40 | 97.70 | 25900.20 | 9.66 / 13.36 | 1885.73 | 508.74 / 11375.01 |
| results/r11-iq32-near8k/iq32 | 8192 / 8128 | 408.00 | 101.00 | 19953.56 | 9.69 / 10.87 | 1884.10 | 508.74 / 11375.01 |

TTFT desde el inicio de prefill, excluida la carga inicial. VRAM y GTT son máximos muestreados del dispositivo entero.
