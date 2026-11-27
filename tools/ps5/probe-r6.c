/* Host-only diagnostics: streaming read and dependent randomized cache-line chase. */
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>
static volatile uint64_t sink;
static double now(void) { struct timespec t; if(clock_gettime(CLOCK_MONOTONIC,&t)) abort(); return t.tv_sec+t.tv_nsec*1e-9; }
static uint32_t rng=0x7df72135u;
static uint32_t rnd(void) { rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; }
static void *aligned(size_t n) { void *p=NULL; if(posix_memalign(&p,64,n)) abort(); return p; }
__attribute__((noinline)) static uint64_t read_avx(const uint64_t *p,size_t n) {
    __m256i a=_mm256_setzero_si256(),b=a,c=a,d=a;
    for(size_t i=0;i<n;i+=16) {
        a=_mm256_add_epi64(a,_mm256_load_si256((const __m256i*)(p+i)));
        b=_mm256_add_epi64(b,_mm256_load_si256((const __m256i*)(p+i+4)));
        c=_mm256_add_epi64(c,_mm256_load_si256((const __m256i*)(p+i+8)));
        d=_mm256_add_epi64(d,_mm256_load_si256((const __m256i*)(p+i+12)));
    }
    uint64_t v[4]; _mm256_storeu_si256((__m256i*)v,_mm256_add_epi64(_mm256_add_epi64(a,b),_mm256_add_epi64(c,d)));
    return v[0]+v[1]+v[2]+v[3];
}
static void probe(size_t bytes) {
    size_t n=bytes/8, lines=bytes/64; uint64_t *p=aligned(bytes);
    for(size_t i=0;i<n;i++) p[i]=i+1;
    uint64_t want=(uint64_t)n*(n+1)/2;
    if(read_avx(p,n)!=want) abort();
    size_t reps=(256u<<20)/bytes; if(reps<4) reps=4;
    double start=now();
    for(size_t r=0;r<reps;r++) { /* Prevent loop-invariant hoisting of the read. */
        __asm__ __volatile__("" ::: "memory"); sink=read_avx(p,n);
    }
    double elapsed=now()-start; if(elapsed<=0 || sink!=want) abort();
    uint32_t *order=malloc(lines*sizeof(*order)); if(!order) abort();
    for(size_t i=0;i<lines;i++) order[i]=(uint32_t)i;
    for(size_t i=lines-1;i>0;i--) { size_t j=rnd()%(i+1); uint32_t x=order[i]; order[i]=order[j]; order[j]=x; }
    for(size_t i=0;i<lines;i++) p[(size_t)order[i]*8]=(uint64_t)order[(i+1)%lines]*8;
    size_t ix=(size_t)order[0]*8, initial=ix;
    for(size_t i=0;i<lines;i++) ix=p[ix];
    if(ix!=initial) abort();
    size_t steps=lines*4; if(steps<2000000) steps=2000000;
    start=now(); for(size_t i=0;i<steps;i++) ix=p[ix]; double latency=now()-start; sink=ix;
    if(ix!=(size_t)order[steps%lines]*8 || latency<=0) abort();
    printf("%zu,%.3f,%.3f\n",bytes,(double)bytes*reps/elapsed/1e9,latency*1e9/steps);
    free(order); free(p);
}
int main(void) {
    puts("working_set_bytes,cpu_single_thread_read_GB_s,dependent_load_ns");
    const size_t sizes[]={16u<<10,256u<<10,4u<<20,32u<<20,64u<<20};
    for(size_t i=0;i<sizeof(sizes)/sizeof(sizes[0]);i++) probe(sizes[i]);
    return 0;
}
