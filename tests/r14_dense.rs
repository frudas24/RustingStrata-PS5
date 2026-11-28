//! Physical dense integer candidate; compare against routed full-K oracle and separate baseline.
#![cfg(feature = "ps5-r2")]
mod common;
use std::io::Write;
use rusting_strata::{backend::Backend, quant::GgmlType, vulkan::{Vk, spv_smem::shared_bytes}, weights::Tensor};

#[test]
fn shared_bound() {
    for spec in [[0,13,176,0,0],[0,14,210,0,0]] {
        let a=shared_bytes("gemmi_dp", &spec, 0).unwrap();
        let b=shared_bytes("gemmi_dp_ps5_32", &spec, 0).unwrap();
        let c=shared_bytes("gemmi_dp_ps5_64", &spec, 0).unwrap();
        assert!(b<c && c<a && a<=64*1024, "LDS {a}/{b}/{c}");
        eprintln!("R14 LDS type={} base={a} tile32={b} tile64={c}",spec[1]);
    }
}

#[test]
#[ignore="physical Vulkan required"]
fn fingerprint() {
    let g=Vk::new(0).expect("physical Vulkan required");
    let path=std::env::var("PS5_R14_GPU_FILE").expect("fingerprint path");
    let mut out=std::io::BufWriter::new(std::fs::OpenOptions::new().write(true).create_new(true).open(path).unwrap());
    let mut cases=0usize; let mut values=0usize;
    for (ty,raw,shapes) in [
        (GgmlType::Q5K,&include_bytes!("fixtures/Q5_K.raw")[..],[(512,2048),(8192,2048),(4096,2048)]),
        (GgmlType::Q6K,&include_bytes!("fixtures/Q6_K.raw")[..],[(2048,512),(2048,4096),(512,2048)])
    ] {
        assert!(g.int8_gemm(ty));
        let mut geometry=Vec::new();
        for cols in [256usize,512,2048] { for rows in [19usize,64,65] {
            for n in [1usize,31,32,33,63,64,65,128,129] { for act in [0,1,2] { geometry.push((rows,cols,n,act)); } }
        }}
        for (rows,cols) in shapes { for n in [33,129] { geometry.push((rows,cols,n,0)); } }
        for (rows,cols,n,act) in geometry {
            let rb=ty.row_bytes(cols as u64);
            let data:Vec<u8>=raw.iter().copied().cycle().take(rows*rb).collect();
            let w=g.upload(&Tensor{ty,rows,cols,data:&data,scale:1.}).unwrap();
            let xs=cols+32; let ys=rows+3;
            let x=common::up(&g,&(0..(n+2)*xs).map(|i|((i*37%251) as f32-125.)/127.).collect::<Vec<_>>());
            let mut initial=vec![0.125f32;(n+2)*ys];
            if act==0 { for r in 1..=n { initial[r*ys..r*ys+rows].fill(f32::NAN); } }
            let y=common::up(&g,&initial); let oracle=common::up(&g,&initial);
            let xp=x.addr()+(4*xs) as u64;
            g.gemm_dense(&w,n,xp,xs,y.addr()+(4*ys) as u64,ys,act).unwrap();
            let mut tiles=Vec::new();
            for start in (0..n).step_by(128) { tiles.extend([w.ptr,start as u64|(((n-start).min(128) as u64)<<32)]); }
            let list=common::up(&g,&tiles);
            g.gemm(ty,rows,cols,list.addr(),tiles.len()/2,xp,xs,n,0,oracle.addr()+(4*ys) as u64,ys,act).unwrap();
            let v=common::down::<f32>(&g,&y); let expected=common::down::<f32>(&g,&oracle);
            assert_eq!(v.len(),initial.len()); assert_eq!(expected.len(),v.len());
            assert!(v.iter().zip(&expected).all(|(a,b)|a.to_bits()==b.to_bits()),"{ty:?} {rows}x{cols} n={n} act={act}");
            for (r,row) in v.chunks_exact(ys).enumerate() {
                assert!(row.iter().all(|x|x.is_finite()));
                assert!(row[rows..].iter().all(|&x|x==0.125));
                if r==0 || r==n+1 { assert!(row.iter().all(|&x|x==0.125)); }
            }
            for val in &v { out.write_all(&val.to_bits().to_le_bytes()).unwrap(); }
            cases+=1; values+=v.len();
        }
    }
    assert_eq!(cases,498);
    out.flush().unwrap();
    eprintln!("R14 dense GPU parity: {cases} cases {values} values");
}
