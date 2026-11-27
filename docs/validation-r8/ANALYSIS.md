# Análisis físico

## Puertas del bundle

- La compilación nativa en la PS5 usó Rust 1.100.0, Shaderc 2026.1 y SPIRV-Tools 2026.1. El log identifica el commit exacto y registra 3m29s.
- test-r7 pasó atención vacía en las dos rutas y el fixture del mailbox MoE.
- test-r2 salió 101 al inicializar la primera variante acelerada. El baseline de la puerta y la prueba LDS se ejecutaron. RADV declara falso para dot INT8 empaquetado acelerado. La falta de capacidad impidió evaluar las variantes que la requieren.
- R5 ejecutó 32 fingerprints en las ocho parejas estándar y 64 en las dieciséis parejas K/V. Se preservaron los hashes, las variantes despachadas y los resultados bit a bit.
- GEMM-64 pasa la puerta física. El log exige el marcador del kernel candidato en su brazo y la igualdad de los buffers completos. El pase de paridad es una condición previa para medir rendimiento.

## Partes del trabajo GEMM

Dentro de gemmi.glsl, ldx lleva activaciones cuantizadas y metadata a LDS, mientras ldw lee y desempaqueta los bloques de peso usando qw. La reducción consume valores de LDS, calcula productos punto INT8 empaquetados y los combina con escalas y términos mínimos en scale_add. La carga de la iteración siguiente y los productos de la actual se intercalan en el mismo kernel. El compilador puede solapar latencias de memoria y aritmética. Los timestamps Vulkan por dispatch miden ese conjunto. Las estadísticas de pipeline cuentan propiedades del código y recursos, y por sí solas no dan fracciones de tiempo de ejecución. La cuantización de activaciones quant_q8 usa un kernel separado y se mide de forma independiente.

A igual matriz, el tile de 64 reduce LDS pero aumenta los grupos de posiciones. Para 512 posiciones, se programan dos veces más tiles en M que con 128. Cada grupo lógico vuelve a leer y decodificar su franja de peso. GDDR6, cachés y latencia de instrucciones determinan el coste físico. El cálculo de grupos no es un medidor de bytes del controlador. Los logs de profiling se generan en procesos distintos de los A/B de rendimiento. No se alteró ninguna barrera.

## Método R4

El parche de investigación observa una fila realmente enrutada al shader GPU y evalúa el mismo experto con el mismo vector hidden en CPU. Lee gate previo a fusión, up como salida separada, activación SiLU×up, down usando la activación GPU, y down mediante la ruta CPU completa. El brazo híbrido también comprueba una fila escrita por el scatter CPU. El cálculo f64 auxiliar de 16 filas usa los mismos pesos descomprimidos en ambos brazos para contrastar solo agrupación de cuantización de la activación.

La CPU usa bloques Q8K de 256 y escala derivada del valor de mayor magnitud con signo. El quant_q8 normal de Vulkan usa bloques de 32 y escala positiva max(abs(x))/127. La variante de numerics llama usa otra agrupación, pero también cambia otras rutas del motor. Después del dot, la GPU usa acumulación float precisa y FMA en scale_add, además de término de mínimo cuando aplica. CPU agrupa los productos con los kernels int_dot del tipo y aplica Tensor.scale. La ruta Vulkan de tiles expertos pasa escala externa 1,0. El estudio registra las escalas observadas para distinguir esa posibilidad de la diferencia de bloques. El orden de suma y la implementación de SiLU también pueden producir diferencias pequeñas. Hacer lecturas intermedias sincroniza el diagnóstico y lo excluye de toda evaluación de velocidad.

## Campañas de GEMM-64

La puerta de buffers validó 720 casos y 6.033.510 valores f32 por brazo. En 0,6B el primer intento falló la puerta de estabilidad en prosa. Se ejecutó un reintento idéntico tras la campaña 4B, conservando el intento original. El reintento calificó en prosa y código con ganancias geométricas 3,83 % y 3,86 %. El prefill a 128 pasó de 1817,8 a 2451,1 tok/s en prosa, con decode de 387,90 a 387,55 tok/s. La puerta de peor caso fue 99,62 % en prosa y 99,45 % en código.

En 4B completaron ambos bloques A/B/B/A con paridad greedy y top5. El prefill de 128 mejoró de 344,1 a 395,35 tok/s en prosa y de 345,0 a 397,55 en código. Los agregados fueron 2,07 % y 1,83 %. El candidato no pasó el umbral original de ganancia del 3 % para este modelo. El código y las puertas originales quedaron intactos. No se generó ningún perfil de producción.

## Comparación R6U1/R7

Se fija Q8 en K y V, MTP apagado, Vulkan, chunk 512, seis hilos, 32 tokens de generación y GEMM-128. Los dos binarios usan el mismo GGUF y mismos márgenes. Las profundidades programadas son 8.160, 32.736 y 65.024 sobre contextos 8.192, 32.768 y 65.536. 8K usa tres repeticiones por binario. 32K y 64K usan una pasada exploratoria por binario para acotar la duración de prefill. El orden es base seguido de R7 en cada contexto.

## Timestamps R7 y perfiles del driver

| Modelo | GEMM-128 acumulado | GEMM-64 acumulado | Cambio | quant_q8 128 / 64 |
|---|---:|---:|---:|---:|
| 0,6B | 2.105,91 ms | 2.002,14 ms | −4,93 % | 31,41 / 31,41 ms |
| 4B | 16.046,16 ms | 15.749,89 ms | −1,85 % | 91,10 / 90,16 ms |

Las campañas instrumentadas usaron prompts 128/512/2048 con 16 tokens de salida. Se observaron 3.528 dispatches GEMM fuera de graphs por brazo en 0,6B y 4.536 en 4B. Son sumas de intervalos GPU capturados, con el alcance de barreras del profiler de R7, separadas del rendimiento sin instrumentación. quant_q8 representa aproximadamente 1,5 % del subtotal GEMM+quant_q8 en 0,6B y 0,6 % en 4B. Es la cuantización de activaciones. El desempaquetado de pesos permanece dentro de GEMM.

En la especialización Q4_K observada, RADV reportó LDS de 40.960 a 27.648 bytes, 4 subgrupos/SIMD y 256 VGPR en ambos tiles, sin spills. Las estadísticas estáticas cambiaron de 6.517 a 3.633 instrucciones y de 112 a 64 VMEM. El tamaño de trabajo por grupo también cambia. El perfil publicado no separa ciclos de acceso de memoria, descompresión y dot dentro de gemmi; los porcentajes exactos de esas tres partes quedan abiertos.

## Resultado R4 por etapa

La primera divergencia muestreada apareció en gate de la capa 0, antes de aplicar SiLU o copiar resultados CPU. En el brazo GPU-only para experto 2, token 1, el máximo absoluto fue 0,016440451 en gate y 0,013765991 en up. En el brazo half para experto 18, token 4, gate dio 0,016881928 y up 0,014676854. Dentro de cada comparación ambas rutas usan el mismo experto y hidden. Los expertos muestreados entre brazos difieren por la selección del planificador.

En half, la referencia de pesos descomprimidos y bloques de 32 se acercó más a la GPU que la referencia Q8K-256 en 924/960 muestras. En GPU-only fueron 927/960. En half las medianas de máximos absolutos por muestra fueron 0,006936 y 0,013012, respectivamente. Las escalas por tensores se registran en los logs. Las diferencias al mantener fija la activación GPU en down indican que el error no se limita a gate/up. El desvío restante puede incluir descompresión de pesos, escalas, redondeo y reducción. El experimento no separa todas esas contribuciones.

El readback de scatter coincidió bit a bit con la fila CPU escrita en las 960 muestras del brazo half. Eso descarta una discrepancia sobre esas escrituras muestreadas. La instrumentación añade esperas y mide errores locales con el mismo input. No atribuye automáticamente la divergencia final de logits a una única operación. R4 permanece apagado fuera del ensayo de investigación.

A 8K, los dos binarios reservaron 85 MiB de KV y 10.240 expertos residentes. Las tres huellas greedy y los top-5 completos fueron idénticos. Prefill pasó de 313,5 a 317,0 tok/s. La media decode pasó de 91,5 a 101,4, con máximos 102,1 y 102,0 y mínimo de 70,3 en R6U1. La media de decode es sensible a esa repetición lenta y no demuestra por sí sola un cambio estable.

A 32K, con las mismas 10.240 filas de expertos residentes y el mismo KV de 340 MiB, prefill fue 205,2 tok/s en ambos binarios. Decode fue 82,9 en R6U1 y 81,8 en R7. La secuencia greedy y los top-5 coincidieron con delta máximo de logp 0 en la huella. Una observación por brazo no caracteriza variabilidad entre corridas.

La escala externa fue 1,0 en cada tensor muestreado de ambos brazos. Gate y up fueron IQ2_XS en 936 casos e IQ3_XXS en 24 por brazo. Down fue IQ3_XXS en 864 e IQ4_XS en 96 por brazo. Por tanto, la escala externa aplicada como 1,0 en el GEMM enrutado no explica la diferencia en este conjunto. Los oráculos usan redondeo y acumulación independientes en CPU para aproximar la descompresión de pesos, con el mismo orden en ambas variantes de activación.
