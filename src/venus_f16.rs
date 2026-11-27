//! Experimental CPU F16 GEMM. Inspired by UI-Venus' 3-row / 4-token F16C register tile.
//! Only full tiles use the new accumulation order; all edges retain the reference dot.
use crate::weights::{Tensor, dot};
use crate::quant::GgmlType;
use rayon::prelude::*;

pub(crate) fn supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    { std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c") }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

pub(crate) fn eligible(t: &Tensor<'_>, n: usize) -> bool {
    t.ty == GgmlType::F16 && t.scale == 1.0 && t.rows >= 3 && n >= 4 && t.cols >= 8 && t.cols % 8 == 0
        && t.rows.checked_mul(t.cols).and_then(|v| v.checked_mul(2)) == Some(t.data.len()) && supported()
}

/// Returns false without touching out if the candidate cannot run.
pub(crate) fn try_matmul(t: &Tensor<'_>, x: &[f32], n: usize, out: &mut [f32]) -> bool {
    if !eligible(t, n) || t.cols.checked_mul(n) != Some(x.len()) || t.rows.checked_mul(n) != Some(out.len()) { return false; }
    #[cfg(target_arch = "x86_64")]
    {
        let rb = t.cols * 2;
        let mut by_row = vec![0f32; out.len()];
        // Rayon owns disjoint output rows. A group of three rows shares four activation loads.
        by_row.par_chunks_mut(3 * n).enumerate().for_each(|(group, ys)| {
            let r = group * 3;
            let nr = ys.len() / n;
            let mut start = 0;
            if nr == 3 {
                let w = &t.data[r * rb..(r + 3) * rb];
                for tok in (0..n - n % 4).step_by(4) {
                    let inputs = &x[tok * t.cols..(tok + 4) * t.cols];
                    // SAFETY: eligibility establishes features, nonzero 8-multiple cols; both slices
                    // contain exactly the full tile. Loads are unaligned and never read past a row.
                    let z = unsafe { tile_3x4(w, inputs, t.cols) };
                    for i in 0..3 { ys[i * n + tok..i * n + tok + 4].copy_from_slice(&z[i * 4..i * 4 + 4]); }
                }
                start = n - n % 4;
            }
            if start < n {
                let mut scratch = vec![0f32; t.cols];
                for i in 0..nr {
                    t.row(r + i, &mut scratch);
                    for tok in start..n { ys[i * n + tok] = dot(&scratch, &x[tok * t.cols..(tok + 1) * t.cols]); }
                }
            }
        });
        out.par_chunks_mut(t.rows).enumerate().for_each(|(tok, ys)| {
            for (r, y) in ys.iter_mut().enumerate() { *y = by_row[r * n + tok]; }
        });
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,f16c")]
unsafe fn tile_3x4(w: &[u8], x: &[f32], cols: usize) -> [f32; 12] {
    use std::arch::x86_64::*;
    // Explicit register tile, not a weight expansion. FMA/reduction differ from weights::dot.
    unsafe {
        let mut a00 = _mm256_setzero_ps();
        let mut a01 = _mm256_setzero_ps();
        let mut a02 = _mm256_setzero_ps();
        let mut a03 = _mm256_setzero_ps();
        let mut a10 = _mm256_setzero_ps();
        let mut a11 = _mm256_setzero_ps();
        let mut a12 = _mm256_setzero_ps();
        let mut a13 = _mm256_setzero_ps();
        let mut a20 = _mm256_setzero_ps();
        let mut a21 = _mm256_setzero_ps();
        let mut a22 = _mm256_setzero_ps();
        let mut a23 = _mm256_setzero_ps();
        for i in (0..cols).step_by(8) {
            let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(2 * (0 * cols + i)).cast()));
            let w1 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(2 * (1 * cols + i)).cast()));
            let w2 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(2 * (2 * cols + i)).cast()));
            let x0 = _mm256_loadu_ps(x.as_ptr().add(0 * cols + i));
            a00 = _mm256_fmadd_ps(w0, x0, a00);
            a10 = _mm256_fmadd_ps(w1, x0, a10);
            a20 = _mm256_fmadd_ps(w2, x0, a20);
            let x1 = _mm256_loadu_ps(x.as_ptr().add(1 * cols + i));
            a01 = _mm256_fmadd_ps(w0, x1, a01);
            a11 = _mm256_fmadd_ps(w1, x1, a11);
            a21 = _mm256_fmadd_ps(w2, x1, a21);
            let x2 = _mm256_loadu_ps(x.as_ptr().add(2 * cols + i));
            a02 = _mm256_fmadd_ps(w0, x2, a02);
            a12 = _mm256_fmadd_ps(w1, x2, a12);
            a22 = _mm256_fmadd_ps(w2, x2, a22);
            let x3 = _mm256_loadu_ps(x.as_ptr().add(3 * cols + i));
            a03 = _mm256_fmadd_ps(w0, x3, a03);
            a13 = _mm256_fmadd_ps(w1, x3, a13);
            a23 = _mm256_fmadd_ps(w2, x3, a23);
        }
        let mut z = [0f32; 12];
        let s = _mm_add_ps(_mm256_castps256_ps128(a00), _mm256_extractf128_ps::<1>(a00));
        let s = _mm_hadd_ps(s, s);
        z[0] = _mm_cvtss_f32(_mm_hadd_ps(s, s));
        let s = _mm_add_ps(_mm256_castps256_ps128(a01), _mm256_extractf128_ps::<1>(a01));
        let s = _mm_hadd_ps(s, s);
        z[1] = _mm_cvtss_f32(_mm_hadd_ps(s, s));
        let s = _mm_add_ps(_mm256_castps256_ps128(a02), _mm256_extractf128_ps::<1>(a02));
        let s = _mm_hadd_ps(s, s);
        z[2] = _mm_cvtss_f32(_mm_hadd_ps(s, s));
        let s = _mm_add_ps(_mm256_castps256_ps128(a03), _mm256_extractf128_ps::<1>(a03));
        let s = _mm_hadd_ps(s, s);
        z[3] = _mm_cvtss_f32(_mm_hadd_ps(s, s));
        let s = _mm_add_ps(_mm256_castps256_ps128(a10), _mm256_extractf128_ps::<1>(a10));
        let s = _mm_hadd_ps(s, s);
        z[4] = _mm_cvtss_f32(_mm_hadd_ps(s, s));
        let s = _mm_add_ps(_mm256_castps256_ps128(a11), _mm256_extractf128_ps::<1>(a11));
        let s = _mm_hadd_ps(s, s);
        z[5] = _mm_cvtss_f32(_mm_hadd_ps(s, s));
        let s = _mm_add_ps(_mm256_castps256_ps128(a12), _mm256_extractf128_ps::<1>(a12));
        let s = _mm_hadd_ps(s, s);
        z[6] = _mm_cvtss_f32(_mm_hadd_ps(s, s));
        let s = _mm_add_ps(_mm256_castps256_ps128(a13), _mm256_extractf128_ps::<1>(a13));
        let s = _mm_hadd_ps(s, s);
        z[7] = _mm_cvtss_f32(_mm_hadd_ps(s, s));
        let s = _mm_add_ps(_mm256_castps256_ps128(a20), _mm256_extractf128_ps::<1>(a20));
        let s = _mm_hadd_ps(s, s);
        z[8] = _mm_cvtss_f32(_mm_hadd_ps(s, s));
        let s = _mm_add_ps(_mm256_castps256_ps128(a21), _mm256_extractf128_ps::<1>(a21));
        let s = _mm_hadd_ps(s, s);
        z[9] = _mm_cvtss_f32(_mm_hadd_ps(s, s));
        let s = _mm_add_ps(_mm256_castps256_ps128(a22), _mm256_extractf128_ps::<1>(a22));
        let s = _mm_hadd_ps(s, s);
        z[10] = _mm_cvtss_f32(_mm_hadd_ps(s, s));
        let s = _mm_add_ps(_mm256_castps256_ps128(a23), _mm256_extractf128_ps::<1>(a23));
        let s = _mm_hadd_ps(s, s);
        z[11] = _mm_cvtss_f32(_mm_hadd_ps(s, s));
        z
    }
}

/// Single-token F16 path derived from Venus' four-accumulator dotF16AVX.
/// Kept independently gated from the 3x4 kernel: this reaches ordinary CPU text decode.
pub(crate) fn try_matvec(t: &Tensor<'_>, x: &[f32], out: &mut [f32]) -> bool {
    if t.ty != GgmlType::F16 || t.scale != 1.0 || t.cols < 8 || t.cols % 8 != 0 || t.rows == 0
        || t.cols != x.len() || t.rows != out.len() || !supported()
        || t.rows.checked_mul(t.cols).and_then(|n| n.checked_mul(2)) != Some(t.data.len()) { return false; }
    #[cfg(target_arch="x86_64")]
    {
        let rb = 2*t.cols;
        out.par_chunks_mut(16).enumerate().for_each(|(b, ys)| {
            for (i,y) in ys.iter_mut().enumerate() {
                let r=b*16+i;
                // SAFETY: the F16 view, activation lengths and CPU features are checked above.
                *y=unsafe { dot_f16(&t.data[r*rb..(r+1)*rb],x) };
            }
        });
        true
    }
    #[cfg(not(target_arch="x86_64"))]
    { false }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx,fma,f16c")]
unsafe fn dot_f16(w: &[u8], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut a0=_mm256_setzero_ps(); let mut a1=a0; let mut a2=a0; let mut a3=a0;
        let mut i=0;
        macro_rules! step { ($a:ident,$j:expr) => {{
            let j=$j;
            let v=_mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(2*j).cast()));
            $a=_mm256_fmadd_ps(v,_mm256_loadu_ps(x.as_ptr().add(j)),$a);
        }}; }
        while x.len()-i>=32 {
            step!(a0,i); step!(a1,i+8); step!(a2,i+16); step!(a3,i+24); i+=32;
        }
        a0=_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0,a1),a2),a3);
        while i<x.len() { step!(a0,i); i+=8; }
        let s=_mm_add_ps(_mm256_castps256_ps128(a0),_mm256_extractf128_ps::<1>(a0));
        let s=_mm_hadd_ps(s,s);
        _mm_cvtss_f32(_mm_hadd_ps(s,s))
    }
}

/// BF16 is already the upper 16 bits of IEEE FP32. Zen 2 can expand eight
/// values with AVX2 zero-extension + shift, without converting to F16 or
/// requiring AVX512_BF16. Separate flag: does not change the F16 path.
pub(crate) fn try_bf16_matvec(t: &Tensor<'_>, x: &[f32], out: &mut [f32]) -> bool {
    if t.ty != GgmlType::BF16 || t.scale != 1.0 || t.cols < 8 || t.cols % 8 != 0 || t.rows == 0
        || t.cols != x.len() || t.rows != out.len()
        || t.rows.checked_mul(t.cols).and_then(|n| n.checked_mul(2)) != Some(t.data.len()) { return false; }
    #[cfg(target_arch="x86_64")]
    {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") { return false; }
        let rb=2*t.cols;
        out.par_chunks_mut(16).enumerate().for_each(|(b, ys)| {
            for (i,y) in ys.iter_mut().enumerate() {
                let r=b*16+i;
                // SAFETY: features, row length, input and output sizes checked above.
                *y=unsafe { dot_bf16(&t.data[r*rb..(r+1)*rb],x) };
            }
        });
        true
    }
    #[cfg(not(target_arch="x86_64"))]
    { false }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2,fma")]
unsafe fn dot_bf16(w: &[u8], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut a0=_mm256_setzero_ps(); let mut a1=a0; let mut a2=a0; let mut a3=a0;
        let mut i=0;
        macro_rules! step { ($a:ident,$j:expr) => {{
            let j=$j;
            let bits=_mm256_cvtepu16_epi32(_mm_loadu_si128(w.as_ptr().add(2*j).cast()));
            let v=_mm256_castsi256_ps(_mm256_slli_epi32::<16>(bits));
            $a=_mm256_fmadd_ps(v,_mm256_loadu_ps(x.as_ptr().add(j)),$a);
        }}; }
        while x.len()-i>=32 { step!(a0,i); step!(a1,i+8); step!(a2,i+16); step!(a3,i+24); i+=32; }
        a0=_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0,a1),a2),a3);
        while i<x.len() { step!(a0,i); i+=8; }
        let s=_mm_add_ps(_mm256_castps256_ps128(a0),_mm256_extractf128_ps::<1>(a0));
        let s=_mm_hadd_ps(s,s);
        _mm_cvtss_f32(_mm_hadd_ps(s,s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn venus_f16_shapes_and_reference() {
        if !supported() { eprintln!("SKIP: requires AVX/FMA/F16C"); return; }
        for rows in [3, 4, 5, 6, 17] {
            for cols in [8, 16, 24, 128, 1032] {
                for n in [4, 5, 7, 8, 13] {
                    // Odd byte address: exercise unaligned GGUF views.
                    let mut data = vec![0x55];
                    for i in 0..rows * cols { data.extend(f16::from_f32(((i * 71 % 197) as f32 - 98.) / 71.).to_le_bytes()); }
                    let t = Tensor { ty: GgmlType::F16, cols, rows, data: &data[1..], scale: 1.0 };
                    let x: Vec<_> = (0..n * cols).map(|i| ((i * 41 % 173) as f32 - 86.) / 79.).collect();
                    let mut y = vec![f32::NAN; rows * n];
                    assert!(try_matmul(&t, &x, n, &mut y));
                    let mut w = vec![0.; cols];
                    for r in 0..rows {
                        t.row(r, &mut w);
                        for tok in 0..n {
                            let input = &x[tok * cols..(tok + 1) * cols];
                            let actual = y[tok * rows + r];
                            let want = dot(&w, input);
                            let abs_sum: f64 = w.iter().zip(input).map(|(&a,&b)| (a as f64 * b as f64).abs()).sum();
                            let exact: f64 = w.iter().zip(input).map(|(&a,&b)| a as f64 * b as f64).sum();
                            assert!(actual.is_finite() && (actual as f64 - exact).abs() <= 2e-6 * abs_sum + 1e-6);
                            if r >= rows - rows % 3 || tok >= n - n % 4 { assert_eq!(actual.to_bits(), want.to_bits()); }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn venus_f16_matvec_reference() {
        if !supported() { eprintln!("SKIP: requires AVX/FMA/F16C"); return; }
        for cols in [8,16,24,32,40,56,128,1032] {
            let rows=19;
            let mut raw=vec![0u8];
            for i in 0..rows*cols { raw.extend(f16::from_f32(((i*71%197) as f32-98.)/71.).to_le_bytes()); }
            let t=Tensor{ty:GgmlType::F16,cols,rows,data:&raw[1..],scale:1.};
            let x:Vec<_>=(0..cols).map(|i| ((i*41%173) as f32-86.)/79.).collect();
            let mut y=vec![f32::NAN;rows]; assert!(try_matvec(&t,&x,&mut y));
            let mut w=vec![0.;cols];
            for (r,&actual) in y.iter().enumerate() {
                t.row(r,&mut w);
                let exact:f64=w.iter().zip(&x).map(|(&a,&b)|a as f64*b as f64).sum();
                let abs_sum:f64=w.iter().zip(&x).map(|(&a,&b)|(a as f64*b as f64).abs()).sum();
                assert!(actual.is_finite() && (actual as f64-exact).abs()<=2e-6*abs_sum+1e-6);
            }
        }
    }

    #[test]
    fn venus_bf16_matvec_reference_and_fallback() {
        for cols in [8,16,24,32,40,56,128,1032] {
            let rows=19;
            let mut raw=vec![0u8];
            for i in 0..rows*cols { raw.extend(half::bf16::from_f32(((i*71%197) as f32-98.)/71.).to_le_bytes()); }
            let t=Tensor{ty:GgmlType::BF16,cols,rows,data:&raw[1..],scale:1.};
            let x:Vec<_>=(0..cols).map(|i| ((i*41%173) as f32-86.)/79.).collect();
            let mut y=vec![17.;rows];
            if try_bf16_matvec(&t,&x,&mut y) {
                let mut w=vec![0.;cols];
                for (r,&actual) in y.iter().enumerate() {
                    t.row(r,&mut w);
                    let exact:f64=w.iter().zip(&x).map(|(&a,&b)|a as f64*b as f64).sum();
                    let abs_sum:f64=w.iter().zip(&x).map(|(&a,&b)|(a as f64*b as f64).abs()).sum();
                    assert!(actual.is_finite() && (actual as f64-exact).abs()<=2e-6*abs_sum+1e-6);
                }
            } else { assert_eq!(y,vec![17.;rows]); }
            for bad in [Tensor{scale:2.,..t},Tensor{ty:GgmlType::F16,..t},Tensor{data:&t.data[..t.data.len()-1],..t}] {
                y.fill(17.);
                assert!(!try_bf16_matvec(&bad,&x,&mut y));
                assert_eq!(y,vec![17.;rows]);
            }
        }
    }

    #[test]
    fn venus_f16_rejects_without_writing() {
        let data = [0u8; 48];
        let base = Tensor { ty: GgmlType::F16, cols: 8, rows: 3, data: &data, scale: 1.0 };
        for t in [Tensor { scale: 2., ..base }, Tensor { ty: GgmlType::F32, ..base }, Tensor { cols: 7, ..base }, Tensor { data: &data[..47], ..base }] {
            let mut y = [17.; 12];
            assert!(!try_matmul(&t, &[0.;32], 4, &mut y));
            assert_eq!(y, [17.;12]);
        }
    }

    #[test]
    #[ignore = "requires a local F16 GGUF; see test-venus.sh"]
    fn venus_f16_real_gguf() {
        assert!(supported(), "AVX/FMA/F16C required");
        assert_ne!(std::env::var("STRATA_PS5_VENUS_F16").as_deref(), Ok("1"), "baseline must be disabled");
        let path = std::env::var("VENUS_F16_MODEL").expect("provide F16 GGUF");
        let gg = crate::gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
        let chosen = std::env::var("VENUS_F16_TENSOR").ok();
        let info = gg.tensors.values().find(|t| t.ty == 1 && t.dims.len() == 2 && t.dims[0] >= 8 && t.dims[0] <= 8192
            && t.dims[0] % 8 == 0 && t.dims[1] >= 3 && chosen.as_ref().is_none_or(|name| name == &t.name))
            .expect("no eligible F16 matrix; choose an F16 GGUF or set VENUS_F16_TENSOR");
        let cols = info.dims[0] as usize;
        let rows = info.dims[1].min(384) as usize;
        let t = Tensor { ty: GgmlType::F16, cols, rows, data: gg.raw_bytes(info, 2*cols*rows).unwrap(), scale:1. };
        for n in [4, 8, 17] {
            let x: Vec<_> = (0..n*cols).map(|i| (((i*41+17)%173) as f32-86.)/79.).collect();
            let mut a = vec![0.;rows*n]; let mut b = a.clone();
            t.matmul(&x,n,&mut a); assert!(try_matmul(&t,&x,n,&mut b));
            let mut w=vec![0.;cols]; let mut worst=0f64;
            for r in 0..rows {
                t.row(r,&mut w);
                for j in 0..n {
                    let v=&x[j*cols..(j+1)*cols];
                    let sum:f64=w.iter().zip(v).map(|(&p,&q)|(p as f64*q as f64).abs()).sum();
                    let d=(a[j*rows+r] as f64-b[j*rows+r] as f64).abs();
                    assert!(a[j*rows+r].is_finite() && b[j*rows+r].is_finite() && d <= 5e-6*sum+1e-6, "F16 parity failed");
                    worst=worst.max(d);
                }
            }
            let mut samples=Vec::new();
            for candidate in [false,true,true,false,false,true,true,false] {
                let now=std::time::Instant::now();
                if candidate { assert!(try_matmul(&t,&x,n,&mut b)); std::hint::black_box(&b); }
                else { t.matmul(&x,n,&mut a); std::hint::black_box(&a); }
                samples.push((candidate,now.elapsed().as_nanos() as u64));
            }
            println!("VENUS_F16_GGUF {}",serde_json::json!({"tensor":info.name,"rows":rows,"cols":cols,"tokens":n,"max_abs":worst,"samples_ns":samples}));
        }
    }

}
