//! Research-only CPU shadow for the Vulkan routed prefill arithmetic contract.
//! Deliberately slow: decode one weight row at a time, not a production kernel.
use crate::{model::Model, ops, quant::GgmlType, weights::Tensor};
use rayon::prelude::*;
use half::f16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Contract { F16, Q8_32 }

pub(crate) fn activation(x: &[f32], kind: Contract) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(!x.is_empty() && x.iter().all(|v| v.is_finite()), "R8 invalid activation");
    let mut out = Vec::with_capacity(x.len());
    match kind {
        Contract::F16 => out.extend(x.iter().map(|&v| f16::from_f32(v).to_f32())),
        Contract::Q8_32 => {
            anyhow::ensure!(x.len() % 32 == 0, "R8 Q8_32 requires complete blocks");
            for block in x.chunks_exact(32) {
                let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
                if amax == 0. { out.extend([0.; 32]); continue; }
                let d = amax / 127.; let inv = 127. / amax;
                anyhow::ensure!(d > 0. && inv.is_finite(), "R8 unsupported Q8 scale");
                out.extend(block.iter().map(|&v| (v * inv).round_ties_even().clamp(-127.,127.) * d));
            }
        }
    }
    anyhow::ensure!(out.iter().all(|v| v.is_finite()), "R8 nonfinite rounded activation");
    Ok(out)
}

/// Reference only. F16 mimics the non-cooperative shader's sequential FP32 FMA.
/// Q8_32 uses dequantized weights and f64 reduction; it does NOT reproduce scale_add's order.
pub(crate) fn projection(t: &Tensor<'_>, x: &[f32], kind: Contract, rows: usize) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(t.scale == 1. && x.len() == t.cols && rows <= t.rows && t.cols > 0
        && t.cols % t.ty.block_elems() == 0 && t.ty.row_bytes(t.cols as u64).checked_mul(t.rows) == Some(t.data.len()),
        "R8 unsupported tensor layout or scale");
    if let Some(y) = crate::ps5_r11::candidate(t,x,kind,rows)? { return Ok(y); }
    let x = activation(x, kind)?;
    let mut y = vec![0.; rows];
    y.par_chunks_mut(16).enumerate().for_each_init(|| vec![0.;t.cols], |buf,(i,ys)| {
        for (j,v) in ys.iter_mut().enumerate() {
            t.row(i*16+j,buf);
            *v = match kind {
                Contract::F16 => buf.iter().zip(&x).fold(0f32, |a,(&w,&x)| f16::from_f32(w).to_f32().mul_add(x,a)),
                Contract::Q8_32 => buf.iter().zip(&x).map(|(&w,&x)| w as f64 * x as f64).sum::<f64>() as f32,
            };
        }
    });
    anyhow::ensure!(y.iter().all(|v| v.is_finite()), "R8 nonfinite projection");
    Ok(y)
}

pub(crate) fn supported(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::IQ2XS | GgmlType::IQ3XXS | GgmlType::IQ4XS)
}

/// Unweighted output rows in original pair order; only used for research-selected R4 misses.
pub(crate) fn expert_rows(m: &Model, il: usize, hs: &[Vec<f32>], pairs: &[(usize,usize,f32)], kinds: [Contract;3]) -> anyhow::Result<Vec<f32>> {
    let c = &m.cfg;
    anyhow::ensure!(c.hadamard.is_none() && m.loras.is_empty(), "R8 excludes Hadamard/LoRA");
    let g = m.w.layer(il,"ffn_gate_exps.weight")?;
    let u = m.w.layer(il,"ffn_up_exps.weight")?;
    let d = m.w.layer(il,"ffn_down_exps.weight")?;
    anyhow::ensure!([g,u,d].iter().all(|t| supported(t.ty) && t.scale == 1.), "R8 untested expert type or scale");
    let mut out = Vec::with_capacity(pairs.len().checked_mul(c.n_embd).ok_or_else(|| anyhow::anyhow!("R8 output overflow"))?);
    for &(tok,e,_) in pairs {
        anyhow::ensure!(tok < hs.len() && e < c.n_expert, "R8 route out of bounds");
        let (g,u,d)=(g.expert(e,c.n_expert),u.expert(e,c.n_expert),d.expert(e,c.n_expert));
        let a=projection(&g,&hs[tok],kinds[0],g.rows)?;
        let b=projection(&u,&hs[tok],kinds[1],u.rows)?;
        anyhow::ensure!(a.len()==b.len() && a.len()==d.cols && d.rows==c.n_embd, "R8 expert dimensions");
        let act: Vec<_>=a.iter().zip(&b).map(|(&a,&b)| ops::silu(a)*b).collect();
        out.extend(projection(&d,&act,kinds[2],d.rows)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn q8_contract() {
        let mut x=vec![0.;64]; x[0]=127.; x[1]=2.5; x[2]=3.5; x[3]=-2.5; x[32]=1.;
        let a=activation(&x,Contract::Q8_32).unwrap();
        assert_eq!(&a[..4], &[127.,2.,4.,-2.]); assert_eq!(a[32],1.);
        assert_eq!(activation(&[0.;32],Contract::Q8_32).unwrap(),vec![0.;32]);
        assert!(activation(&[0.;31],Contract::Q8_32).is_err());
        assert!(activation(&[f32::NAN;32],Contract::Q8_32).is_err());
        assert!(activation(&[f32::MAX;32],Contract::F16).is_err());
        assert!(activation(&[f32::from_bits(1);32],Contract::Q8_32).is_err());
    }
    #[test]
    fn projection_contract() {
        let w: Vec<u8>=[1.0001f32,2.,-1.,0.].iter().flat_map(|v| v.to_le_bytes()).collect();
        let t=Tensor{ty:GgmlType::F32,cols:2,rows:2,data:&w,scale:1.};
        assert_eq!(projection(&t,&[1.,2.],Contract::F16,2).unwrap(),vec![5.,-1.]);
        assert!(projection(&t,&[1.],Contract::F16,2).is_err());
        assert!(projection(&t,&[1.,2.],Contract::F16,3).is_err());
    }
}
