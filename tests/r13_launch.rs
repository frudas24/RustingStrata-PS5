//! Physical regression of Q8 scratch keys across graph record/replay/discard.
#![cfg(feature="ps5-r2")]
mod common;
use rusting_strata::{backend::{Backend,SAME_X},quant::GgmlType,vulkan::Vk,weights::Tensor};

#[test]
#[ignore="requires Vulkan"]
fn q8_graph_cache_lifetime() {
    let g=Vk::new(0).unwrap();
    let (rows,cols,n)=(19usize,256usize,2usize);
    let ty=GgmlType::IQ3XXS;
    assert!(g.int8_gemm(ty));
    let raw=include_bytes!("fixtures/IQ3_XXS.raw");
    let data:Vec<u8>=raw.iter().copied().cycle().take(ty.row_bytes(cols as u64)*rows).collect();
    let w=g.upload(&Tensor{ty,rows,cols,data:&data,scale:1.}).unwrap();
    let host:Vec<f32>=(0..n*cols).map(|i|((i*37%251) as f32-125.)/127.).collect();
    let other:Vec<f32>=host.iter().map(|v| -*v).collect();
    let a=common::up(&g,&host); let b=common::up(&g,&other);
    let y=common::up(&g,&vec![0.0f32;n*rows]);
    let run=|x:u64,act:i32| g.gemm_dense(&w,n,x,cols,y.addr(),rows,act);
    let bits=|| common::down::<f32>(&g,&y).iter().map(|v| {assert!(v.is_finite());v.to_bits()}).collect::<Vec<_>>();
    run(a.addr(),0).unwrap(); let a_ref=bits();
    run(b.addr(),0).unwrap(); let b_ref=bits(); assert_ne!(a_ref,b_ref);
    g.graph(101,&mut || run(a.addr(),0)).unwrap(); assert_eq!(bits(),a_ref);
    g.graph(102,&mut || run(b.addr(),0)).unwrap(); assert_eq!(bits(),b_ref);
    g.graph(101,&mut || panic!("graph was not reused")).unwrap(); assert_eq!(bits(),a_ref);
    g.prequant_gemm_x(&w,n,b.addr(),cols).unwrap();
    run(b.addr(),SAME_X).unwrap(); assert_eq!(bits(),b_ref);
    run(a.addr(),0).unwrap(); assert_eq!(bits(),a_ref);
    assert!(g.graph(103,&mut || {run(b.addr(),0)?; anyhow::bail!("intentional discard")}).is_err());
    g.prequant_gemm_x(&w,n,b.addr(),cols).unwrap();
    run(b.addr(),SAME_X).unwrap(); assert_eq!(bits(),b_ref);
    g.graph_clear().unwrap();

    // More than TMAX rows select the growable scratch, not the fixed small one.
    let big=common::up(&g,&host.iter().copied().cycle().take(20*cols).collect::<Vec<_>>());
    let out=common::up(&g,&vec![0.0f32;20*rows]);
    let large=|n:usize| g.gemm_dense(&w,n,big.addr(),cols,out.addr(),rows,0);
    g.graph(104,&mut || large(18)).unwrap();
    g.sync().unwrap();
    assert!(g.graph(105,&mut || large(19)).is_err(),"must not release a live scratch while recording");
    large(19).unwrap(); // growth outside recording must invalidate the old graph
    let mut rerecorded=false;
    g.graph(104,&mut || {rerecorded=true;large(18)}).unwrap();
    assert!(rerecorded,"old graph retained an address of freed scratch");
    g.sync().unwrap();
    g.graph_clear().unwrap();
    eprintln!("R13 LAUNCH Q8 GRAPH PASS: replay, discard and scratch growth");
}
