/* Independent AVX/FMA/F16C arithmetic/layout check. Does not compile or execute the Rust kernel. */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
static uint32_t seed=1234567;
static uint32_t rng(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
static float sample(void){return ((int)(rng()%20001)-10000)/8192.f;}
static float reduce(__m256 v){__m128 a=_mm_add_ps(_mm256_castps256_ps128(v),_mm256_extractf128_ps(v,1));a=_mm_hadd_ps(a,a);return _mm_cvtss_f32(_mm_hadd_ps(a,a));}
static void tile(const unsigned char *w,const float*x,int n,float*z){
 __m256 a00=_mm256_setzero_ps();
 __m256 a01=_mm256_setzero_ps();
 __m256 a02=_mm256_setzero_ps();
 __m256 a03=_mm256_setzero_ps();
 __m256 a10=_mm256_setzero_ps();
 __m256 a11=_mm256_setzero_ps();
 __m256 a12=_mm256_setzero_ps();
 __m256 a13=_mm256_setzero_ps();
 __m256 a20=_mm256_setzero_ps();
 __m256 a21=_mm256_setzero_ps();
 __m256 a22=_mm256_setzero_ps();
 __m256 a23=_mm256_setzero_ps();
 for(int i=0;i<n;i+=8){
  __m256 w0=_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w+2*(0*n+i))));
  __m256 w1=_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w+2*(1*n+i))));
  __m256 w2=_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w+2*(2*n+i))));
  __m256 x0=_mm256_loadu_ps(x+0*n+i);
  a00=_mm256_fmadd_ps(w0,x0,a00);
  a10=_mm256_fmadd_ps(w1,x0,a10);
  a20=_mm256_fmadd_ps(w2,x0,a20);
  __m256 x1=_mm256_loadu_ps(x+1*n+i);
  a01=_mm256_fmadd_ps(w0,x1,a01);
  a11=_mm256_fmadd_ps(w1,x1,a11);
  a21=_mm256_fmadd_ps(w2,x1,a21);
  __m256 x2=_mm256_loadu_ps(x+2*n+i);
  a02=_mm256_fmadd_ps(w0,x2,a02);
  a12=_mm256_fmadd_ps(w1,x2,a12);
  a22=_mm256_fmadd_ps(w2,x2,a22);
  __m256 x3=_mm256_loadu_ps(x+3*n+i);
  a03=_mm256_fmadd_ps(w0,x3,a03);
  a13=_mm256_fmadd_ps(w1,x3,a13);
  a23=_mm256_fmadd_ps(w2,x3,a23);
 }
 z[0]=reduce(a00);
 z[1]=reduce(a01);
 z[2]=reduce(a02);
 z[3]=reduce(a03);
 z[4]=reduce(a10);
 z[5]=reduce(a11);
 z[6]=reduce(a12);
 z[7]=reduce(a13);
 z[8]=reduce(a20);
 z[9]=reduce(a21);
 z[10]=reduce(a22);
 z[11]=reduce(a23);
}
static float dot_f16(const unsigned char*w,const float*x,int n){
 __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0;
 int i=0;
 #define STEP(a,j) a=_mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w+2*(j)))),_mm256_loadu_ps(x+(j)),a)
 for(;n-i>=32;i+=32){STEP(a0,i);STEP(a1,i+8);STEP(a2,i+16);STEP(a3,i+24);}
 a0=_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0,a1),a2),a3);
 for(;i<n;i+=8){STEP(a0,i);}
 #undef STEP
 return reduce(a0);
}
int main(void){
 if(!__builtin_cpu_supports("avx")||!__builtin_cpu_supports("fma")||!__builtin_cpu_supports("f16c")){fprintf(stderr,"missing SIMD\n");return 1;}
 unsigned char raw[3*2048*2+1];float x[4*2048],z[12];double maxerr=0;
 for(int it=0;it<5000;it++){
  int n=8*(1+rng()%256);unsigned char*w=raw+1;
  for(int i=0;i<3*n;i++){uint16_t h=_cvtss_sh(sample(),0);memcpy(w+2*i,&h,2);}
  for(int i=0;i<4*n;i++)x[i]=sample();
  tile(w,x,n,z);
  for(int r=0;r<3;r++)for(int t=0;t<4;t++){
   double ref=0,abs_sum=0;for(int i=0;i<n;i++){uint16_t h;memcpy(&h,w+2*(r*n+i),2);double v=(double)_cvtsh_ss(h)*x[t*n+i];ref+=v;abs_sum+=fabs(v);}
   double err=fabs(z[4*r+t]-ref);if(err>maxerr)maxerr=err;
   float mv=dot_f16(w+2*r*n,x+t*n,n);
   if(!isfinite(mv)||fabs(mv-ref)>2e-6*abs_sum+1e-6||!isfinite(z[4*r+t])||err>2e-6*abs_sum+1e-6){fprintf(stderr,"FAIL %d %d %d %g\n",it,r,t,err);return 1;}
  }
 }
 printf("PASS 5000 tiles, 60000 GEMM outputs and 60000 GEMV outputs; max absolute error %.9g\n",maxerr);return 0;
}
