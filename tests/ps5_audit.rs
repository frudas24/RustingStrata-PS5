//! Explicit physical Vulkan regressions; no silent skip on missing GPU.
#![cfg(feature="ps5-r2")]
mod common;
use rusting_strata::{backend::Backend, quant::GgmlType, vulkan::Vk, weights::Tensor};

#[test]
#[ignore="requires Vulkan; run test-r11u1-audit.sh --gpu"]
fn moe_duplicate_slots_complete() {
    let g=Vk::new(0).expect("Vulkan required");
    let rows=19usize; let cols=512usize; let ty=GgmlType::IQ2XS;
    let rb=ty.row_bytes(cols as u64);
    let data:Vec<_>=include_bytes!("fixtures/IQ2_XS.raw").iter().copied().cycle().take(2*rows*rb).collect();
    let w=g.upload(&Tensor{ty,rows:2*rows,cols,data:&data,scale:1.}).unwrap();
    // Real distinct activations, one negative slot; repeated experts across up to 32 tokens.
    // 18+ duplicates overflow the old MAX_NM=17 list, while smaller cases exercise dedup.
    for n in [9usize,16,17,18,32] {
        let ids:Vec<i32>=(0..n).map(|i| if i==n-1 && n!=18 {-1} else {0}).collect();
        let input:Vec<f32>=(0..n*cols).map(|i|((i*37+i/cols*11)%251) as f32/127.-1.).collect();
        let x=common::up(&g,&input); let ix=common::up(&g,&ids);
        let sentinel=12345.0f32;
        let y=common::up(&g,&vec![sentinel;n*rows]);
        g.moe_gemv(ty,rows,cols,w.ptr,rows*rb,ix.addr(),n,x.addr(),cols,1,y.addr(),None,0).unwrap();
        let got=common::down::<f32>(&g,&y);
        for i in 0..n {
            let v=&got[i*rows..(i+1)*rows];
            if ids[i]<0 { assert!(v.iter().all(|&a|a==sentinel)); continue; }
            let xx=common::up(&g,&input[i*cols..(i+1)*cols]);
            let id=common::up(&g,&[ids[i]]);
            let yy=common::up(&g,&vec![sentinel;rows]);
            g.moe_gemv(ty,rows,cols,w.ptr,rows*rb,id.addr(),1,xx.addr(),cols,1,yy.addr(),None,0).unwrap();
            let expected=common::down::<f32>(&g,&yy);
            assert!(v.iter().all(|a|a.is_finite() && *a!=sentinel),"missing row n={n} slot={i}");
            assert!(v.iter().zip(expected).all(|(a,b)|a.to_bits()==b.to_bits()),"n={n} slot={i}: dedup changed result");
        }
    }
    eprintln!("AUDIT_MOE_DUPLICATES PASS: 5 shapes, including overflow");
}
