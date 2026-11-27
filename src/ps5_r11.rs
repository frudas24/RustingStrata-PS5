//! Experimental IQ kernels, only wired into the R8 research contract, never production dispatch.
use crate::{quant::{GgmlType, dequant_row, tables::*}, weights::Tensor};
use half::f16;
use rayon::prelude::*;

#[derive(Clone)]
struct Q32 { d: f32, q: [i8;32] }
fn prepare(x: &[f32]) -> anyhow::Result<Vec<Q32>> {
    anyhow::ensure!(x.len()%32 == 0 && x.iter().all(|v| v.is_finite()), "R11 invalid Q8 input");
    x.chunks_exact(32).map(|b| {
        let mx=b.iter().fold(0f32,|a,v|a.max(v.abs()));
        // RADV RCP_D: rounded double division, then f32 multiplication and roundEven.
        let d=(mx as f64/127.) as f32;
        let inv=if mx==0. {0.} else {(127./mx as f64) as f32};
        anyhow::ensure!(inv.is_finite() && (mx==0. || d>0.), "R11 unsupported Q8 scale");
        Ok(Q32{d,q:std::array::from_fn(|i| (b[i]*inv).round_ties_even() as i8)})
    }).collect()
}

// Decode the signed integer code and the single scale per 32 values, as deq.glsl.
fn codes(ty:GgmlType,b:&[u8],ib:usize)->([i8;32],f32) {
    let d=f16::from_le_bytes([b[0],b[1]]).to_f32();
    let mut q=[0i8;32];
    let scale=match ty {
        GgmlType::IQ3XXS=> {
            let aux=u32::from_le_bytes(b[66+4*ib..70+4*ib].try_into().unwrap());
            for l in 0..4 {
                let signs=KSIGNS_IQ2XS[((aux>>(7*l))&127) as usize];
                let g0=IQ3XXS_GRID[b[2+8*ib+2*l] as usize].to_le_bytes();
                let g1=IQ3XXS_GRID[b[3+8*ib+2*l] as usize].to_le_bytes();
                for (j,v) in g0.into_iter().chain(g1).enumerate() {q[l*8+j]=if signs&(1<<j)!=0 {-(v as i8)} else {v as i8};}
            }
            d*(0.5+(aux>>28) as f32)*0.5
        },
        GgmlType::IQ4XS=> {
            let sh=u16::from_le_bytes([b[2],b[3]]);
            let ls=((b[4+ib/2]>>(4*(ib%2)))&15) as i32 | ((((sh>>(2*ib))&3) as i32)<<4);
            for j in 0..16 {let v=b[8+16*ib+j];q[j]=KVALUES_IQ4NL[(v&15) as usize];q[j+16]=KVALUES_IQ4NL[(v>>4) as usize];}
            d*(ls-32) as f32
        },
        _=>unreachable!(),
    };
    (q,scale)
}
fn dot_scalar(w:&[i8;32],x:&[i8;32])->i32 {w.iter().zip(x).map(|(&a,&b)| a as i32*b as i32).sum()}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn dot_avx2(w:&[i8;32],x:&[i8;32])->i32 { unsafe {
    use std::arch::x86_64::*;
    let mut sum=_mm256_setzero_si256();
    // Widen signed values before multiply; no maddubs saturation or -128 sign issue.
    for i in [0,16] {
        let a=_mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(i).cast()));
        let b=_mm256_cvtepi8_epi16(_mm_loadu_si128(x.as_ptr().add(i).cast()));
        sum=_mm256_add_epi32(sum,_mm256_madd_epi16(a,b));
    }
    let mut v=[0i32;8];_mm256_storeu_si256(v.as_mut_ptr().cast(),sum);v.iter().sum()
} }
fn q32_row(ty:GgmlType,row:&[u8],x:&[Q32],dot:fn(&[i8;32],&[i8;32])->i32)->f32 {
    let mut a=0f32;
    for (b,xx) in row.chunks_exact(ty.block_bytes()).zip(x.chunks_exact(8)) {
        for (ib,v) in xx.iter().enumerate() {let (w,d)=codes(ty,b,ib);a=(dot(&w,&v.q) as f32).mul_add(v.d*d,a);}
    }
    a
}
fn layout(t:&Tensor,x:&[f32],rows:usize)->anyhow::Result<usize>{
    anyhow::ensure!(t.scale==1. && t.cols>0 && t.cols%256==0 && x.len()==t.cols && rows<=t.rows,"R11 layout");
    let rb=t.ty.row_bytes(t.cols as u64);
    anyhow::ensure!(rb.checked_mul(t.rows)==Some(t.data.len()),"R11 tensor bytes");Ok(rb)
}
fn q32(t:&Tensor,x:&[f32],rows:usize,simd:bool)->anyhow::Result<Vec<f32>> {
    let rb=layout(t,x,rows)?;
    let x=prepare(x)?;
    let dot:fn(&[i8;32],&[i8;32])->i32=if simd {
        #[cfg(target_arch="x86_64")]
        { if std::is_x86_feature_detected!("avx2") { |w,x|unsafe{dot_avx2(w,x)} } else {dot_scalar} }
        #[cfg(not(target_arch="x86_64"))] {dot_scalar}
    } else {dot_scalar};
    let mut out=vec![0.;rows];
    out.par_iter_mut().enumerate().for_each(|(r,v)| *v=q32_row(t.ty,&t.data[r*rb..(r+1)*rb],&x,dot));
    anyhow::ensure!(out.iter().all(|v|v.is_finite()),"R11 nonfinite Q8 result");Ok(out)
}

// Eight independent output rows per SIMD vector. Each lane preserves the sequential K FMA chain.
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2,fma,f16c")]
unsafe fn iq2_rows(t:&Tensor,x:&[f32],rb:usize,r0:usize,y:&mut[f32]) {unsafe{
    use std::arch::x86_64::*;
    let mut a=_mm256_setzero_ps();
    let mut block=[[0f32;256];8];
    for (ib,xx) in x.chunks_exact(256).enumerate() {
        for (j,b) in block.iter_mut().enumerate().take(y.len()) {
            let o=(r0+j)*rb+ib*74;
            dequant_row(GgmlType::IQ2XS,&t.data[o..o+74],b);
        }
        for (k,&v) in xx.iter().enumerate(){
            let w=_mm256_setr_ps(block[0][k],block[1][k],block[2][k],block[3][k],block[4][k],block[5][k],block[6][k],block[7][k]);
            let w=_mm256_cvtph_ps(_mm256_cvtps_ph::<0>(w));
            a=_mm256_fmadd_ps(w,_mm256_set1_ps(v),a);
        }
    }
    let mut out=[0f32;8];_mm256_storeu_ps(out.as_mut_ptr(),a);y.copy_from_slice(&out[..y.len()]);
}}
fn f16_rows(t:&Tensor,x:&[f32],rows:usize)->anyhow::Result<Option<Vec<f32>>> {
    #[cfg(target_arch="x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c") {
        let rb=layout(t,x,rows)?;
        let x=crate::ps5_r8::activation(x,crate::ps5_r8::Contract::F16)?;
        let mut out=vec![0.;rows];
        out.par_chunks_mut(8).enumerate().for_each(|(r,y)|unsafe{iq2_rows(t,&x,rb,r*8,y)});
        anyhow::ensure!(out.iter().all(|v|v.is_finite()),"R11 nonfinite FP16 result");return Ok(Some(out));
    }
    Ok(None)
}
pub(crate) fn candidate(t:&Tensor,x:&[f32],kind:crate::ps5_r8::Contract,rows:usize)->anyhow::Result<Option<Vec<f32>>> {
    use crate::ps5_r8::Contract;
    if std::env::var_os("STRATA_SCALAR").is_some(){return Ok(None);}
    let out=match (kind,t.ty) {
        (Contract::F16,GgmlType::IQ2XS) if std::env::var("STRATA_PS5_R11_IQ_F16").as_deref()==Ok("1")=>f16_rows(t,x,rows)?,
        (Contract::Q8_32,GgmlType::IQ3XXS|GgmlType::IQ4XS) if std::env::var("STRATA_PS5_R11_IQ_Q32").as_deref()==Ok("1")=>Some(q32(t,x,rows,true)?),
        _=>None,
    };
    if out.is_some() && std::env::var("STRATA_PS5_R11_LOG").as_deref()==Ok("1"){eprintln!("R11 IQ exercised: {:?} {:?} rows={} cols={}",kind,t.ty,rows,t.cols);}
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn integer_extremes(){
        for a in [-128i8,-127,-1,0,1,127] {for b in [-128i8,-127,-1,0,1,127] {
            let w=[a;32];let x=[b;32];
            #[cfg(target_arch="x86_64")] if std::is_x86_feature_detected!("avx2"){assert_eq!(unsafe{dot_avx2(&w,&x)},dot_scalar(&w,&x));}
        }}
    }
    #[test] fn fixture_contracts(){
        for (ty,raw) in [(GgmlType::IQ2XS,include_bytes!("../tests/fixtures/IQ2_XS.raw").as_slice()),(GgmlType::IQ3XXS,include_bytes!("../tests/fixtures/IQ3_XXS.raw").as_slice()),(GgmlType::IQ4XS,include_bytes!("../tests/fixtures/IQ4_XS.raw").as_slice())] {
            let rb=ty.row_bytes(512);let data:Vec<_>=raw.iter().copied().cycle().take(19*rb).collect();let t=Tensor{ty,cols:512,rows:19,data:&data,scale:1.};
            let x:Vec<_>=(0..512).map(|i| ((i*37%251) as f32-125.)/127.).collect();
            if ty==GgmlType::IQ2XS {
                if let Some(y)=f16_rows(&t,&x,19).unwrap(){let xr=crate::ps5_r8::activation(&x,crate::ps5_r8::Contract::F16).unwrap();let mut w=vec![0.;512];for (r,v) in y.iter().enumerate(){t.row(r,&mut w);let a=w.iter().zip(&xr).fold(0f32,|a,(&w,&x)|f16::from_f32(w).to_f32().mul_add(x,a));assert_eq!(a.to_bits(),v.to_bits());}}
            }else{let a=q32(&t,&x,19,false).unwrap();let b=q32(&t,&x,19,true).unwrap();assert_eq!(a.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),b.iter().map(|x|x.to_bits()).collect::<Vec<_>>());}
        }
    }
    #[test]
    #[ignore="requires real Qwen MoE GGUF"]
    fn real_iq_bench(){
        let path=std::env::var("PS5_R11_MODEL").unwrap();let gg=crate::gguf::Gguf::open(std::path::Path::new(&path)).unwrap();let w=crate::weights::Weights::new(&gg).unwrap();let c=crate::config::ModelConfig::from_gguf(&gg).unwrap();
        assert!(c.n_expert>0);let mut seen=[false;3];
        for il in 0..c.n_layer {for name in ["ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"] {
            let Ok(t)=w.layer(il,name) else{continue};let i=match t.ty{GgmlType::IQ2XS=>0,GgmlType::IQ3XXS=>1,GgmlType::IQ4XS=>2,_=>continue};if seen[i]{continue;}seen[i]=true;
            let t=t.expert(0,c.n_expert);let x:Vec<_>=(0..t.cols).map(|i|((i*37%251) as f32-125.)/127.).collect();let mut legacy=vec![0.;t.rows];
            let run=|mode|{let now=std::time::Instant::now();let y=if i==0 {if mode {f16_rows(&t,&x,t.rows).unwrap().expect("F16C required")}else{let xr=crate::ps5_r8::activation(&x,crate::ps5_r8::Contract::F16).unwrap();let mut y=vec![0.;t.rows];y.par_iter_mut().enumerate().for_each_init(||vec![0.;t.cols],|b,(r,v)|{t.row(r,b);*v=b.iter().zip(&xr).fold(0f32,|a,(&w,&x)|f16::from_f32(w).to_f32().mul_add(x,a));});y}}else{q32(&t,&x,t.rows,mode).unwrap()};(y,now.elapsed().as_secs_f64())};
            let _=run(false);let _=run(true);crate::cpu_moe::matvec(&t,&x,&mut legacy);let mut a=vec![];let mut b=vec![];let mut z=vec![];
            for mode in [false,true,true,false,false,true,true,false] {let (y,dt)=run(mode);let (r,_) = run(false);assert!(y.iter().zip(&r).all(|(a,b)|a.to_bits()==b.to_bits()));if mode{b.push(dt)}else{a.push(dt)};let now=std::time::Instant::now();crate::cpu_moe::matvec(&t,&x,&mut legacy);z.push(now.elapsed().as_secs_f64());std::hint::black_box(&legacy);}
            eprintln!("R11_IQ_BENCH {}",serde_json::json!({"layer":il,"tensor":name,"type":t.ty.name(),"rows":t.rows,"cols":t.cols,"bytes":t.data.len(),"reference_seconds":a,"candidate_seconds":b,"legacy_q8k_seconds":z}));
        }}assert!(seen.into_iter().all(|v|v),"required IQ format missing");
    }
}
