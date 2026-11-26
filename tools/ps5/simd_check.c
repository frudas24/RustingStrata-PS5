/* Standalone test of the AVX2 16-lane signed group-sum primitive in cpu_moe.rs.
 * Does not require Rust or a GPU. This validates the reduction, not the engine. */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void sums_avx2(const int8_t *q, int16_t *bs) {
    __m256i packed = _mm256_loadu_si256((const __m256i *)q);
    __m256i pair = _mm256_maddubs_epi16(_mm256_set1_epi8(1), packed);
    __m256i four = _mm256_madd_epi16(pair, _mm256_set1_epi16(1));
    __m256i eight = _mm256_hadd_epi32(four, four);
    __m256i sixteen = _mm256_hadd_epi32(eight, eight);
    bs[0] = (int16_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(sixteen));
    bs[1] = (int16_t)_mm_cvtsi128_si32(_mm256_extracti128_si256(sixteen, 1));
}
int main(void) {
    uint32_t seed = 0x91fda51e;
    for (unsigned k = 0; k < 1000000; ++k) {
        int8_t q[32]; int16_t got[2], ref[2] = {0, 0};
        for (unsigned i = 0; i < 32; ++i) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            q[i] = k == 0 ? -128 : k == 1 ? 127 : (int8_t)(seed & 255);
            ref[i / 16] += q[i];
        }
        sums_avx2(q, got);
        if (memcmp(got, ref, sizeof got) != 0) {
            fprintf(stderr, "mismatch at %u\n", k); return 1;
        }
    }
    puts("PASS: 1000000 AVX2 vectors / 2000000 signed group sums (including -128 and +127 extremes)");
    return 0;
}
