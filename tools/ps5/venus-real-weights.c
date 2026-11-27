/* Independent host probe, not the Rust backend. Compile for Zen 2. */
#define main venus_synthetic_main
#include "venus-f16-check.c"
#undef main
static float bf_dot(const uint16_t*w,const float*x,int n){
 __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0; int i=0;
 #define STEP(a,j) a=_mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(w+(j)))),16)),_mm256_loadu_ps(x+(j)),a)
 for(;n-i>=32;i+=32){STEP(a0,i);STEP(a1,i+8);STEP(a2,i+16);STEP(a3,i+24);}
 a0=_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0,a1),a2),a3);
 for(;i<n;i+=8){STEP(a0,i);}
 #undef STEP
 return reduce(a0);
}
void bf_mv(const uint16_t*w,const float*x,float*y,int rows,int n){for(int r=0;r<rows;r++)y[r]=bf_dot(w+(size_t)r*n,x,n);}
void f16_mv(const uint16_t*w,const float*x,float*y,int rows,int n){for(int r=0;r<rows;r++)y[r]=dot_f16((const unsigned char*)(w+(size_t)r*n),x,n);}
void f16_mm4(const uint16_t*w,const float*x,float*y,int rows,int n){
 int r=0; for(;r+3<=rows;r+=3){float z[12];tile((const unsigned char*)(w+(size_t)r*n),x,n,z);for(int j=0;j<3;j++)for(int t=0;t<4;t++)y[(size_t)t*rows+r+j]=z[j*4+t];}
 for(;r<rows;r++)for(int t=0;t<4;t++)y[(size_t)t*rows+r]=dot_f16((const unsigned char*)(w+(size_t)r*n),x+(size_t)t*n,n);
}
void bf_scalar(const uint16_t*w,const float*x,float*y,int rows,int n){for(int r=0;r<rows;r++){float s=0;for(int i=0;i<n;i++){uint32_t b=(uint32_t)w[(size_t)r*n+i]<<16;float v;memcpy(&v,&b,4);s+=v*x[i];}y[r]=s;}}
