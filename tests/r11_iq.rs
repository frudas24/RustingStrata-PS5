//! Full-output routed IQ2 FP16 shader parity, including indexed and contiguous inputs.
#![cfg(feature="ps5-r2")]
mod common;
use rusting_strata::{backend::Backend,quant::GgmlType,vulkan::{Vk,spv_smem::shared_bytes},weights::Tensor};
#[test]
fn shared_bound(){
    let spec=[0,17,74,0,0];
    let a=shared_bytes("gemm_f_dp",&spec,0).unwrap();
    let b=shared_bytes("gemm_f_dp_ps5_iq32",&spec,0).unwrap();
    assert!(b<a && b<=16*1024,"LDS baseline {a}, candidate {b}");
    eprintln!("R11 IQ LDS baseline={a} candidate={b}");
}
#[test]
#[ignore="requires Vulkan"]
fn fingerprint(){
    let g=Vk::new(0).unwrap();assert!(!g.int8_gemm(GgmlType::IQ2XS));
    let mut all=Vec::new();let mut cases=0;
    for cols in [256usize,512,2048] {for rows in [19usize,64,65] {
        let ty=GgmlType::IQ2XS;let rb=ty.row_bytes(cols as u64);
        let data:Vec<u8>=include_bytes!("fixtures/IQ2_XS.raw").iter().copied().cycle().take(rb*rows).collect();
        let w=g.upload(&Tensor{ty,rows,cols,data:&data,scale:1.}).unwrap();
        let xs=cols+32;let ys=rows+3;let n=132usize;
        let x=common::up(&g,&(0..n*xs).map(|i|((i*37%251) as f32-125.)/127.).collect::<Vec<_>>());
        let ix=common::up(&g,&(0..n).map(|i|((i*17+3)%n) as i32).collect::<Vec<_>>());
        for cnt in [1usize,15,16,17,31,32,33,63,64,65,127,128] {
            // Nonzero base, exactly one tile; use same tile list ABI as routed prefill.
            let tile=common::up(&g,&[w.ptr,2u64|((cnt as u64)<<32)]);
            for indexed in [false,true] {for act in [0,1,2] {
                let y=common::up(&g,&vec![0.125f32;n*ys]);
                g.gemm(ty,rows,cols,tile.addr(),1,x.addr(),xs,n,if indexed{ix.addr()}else{0},y.addr(),ys,act).unwrap();
                let v=common::down::<f32>(&g,&y);
                for (r,row) in v.chunks_exact(ys).enumerate(){
                    assert!(row.iter().all(|v|v.is_finite()));
                    assert!(row[rows..].iter().all(|&v|v==0.125));
                    if r<2||r>=2+cnt{assert!(row.iter().all(|&v|v==0.125));}
                }
                assert!(v.chunks_exact(ys).skip(2).take(cnt).any(|r|r[..rows].iter().any(|&v|v!=0.125)));
                all.extend(v.iter().flat_map(|x|x.to_bits().to_le_bytes()));cases+=1;
            }}
        }
    }}
    assert_eq!(cases,648);
    std::fs::write(std::env::var("PS5_R11_GPU_FILE").unwrap(),&all).unwrap();
    eprintln!("R11 IQ GPU parity: {cases} cases {} values",all.len()/4);
}
