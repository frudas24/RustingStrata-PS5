# Resultado de la campaña

| Contexto | Prompt | n por binario | Prefill R6U1 → R7 | Decode R6U1 → R7 | Greedy / top-5 iguales |
|---:|---:|---:|---:|---:|---:|
| 8192 | 8160 | 3 | 313.5 → 317.0 | 91.5 → 101.4 | 3 / 3 |
| 32768 | 32736 | 1 | 205.2 → 205.2 | 82.9 → 81.8 | 1 / 1 |
| 65536 | 65024 | 1 | 140.4 → 140.5 | 67.4 → 67.3 | 1 / 1 |

Tasas en tokens/s. Los brazos 32K y 64K son exploratorios. El apéndice ANALYSIS.md detalla los límites de la inferencia estadística.

R4 half vs GPU-only produjo hasta 39.75 % de diferencia máxima relativa al rango de logits. La puerta de rango se cumplió en 3/12 vectores y argmax coincidió en 9/12.
