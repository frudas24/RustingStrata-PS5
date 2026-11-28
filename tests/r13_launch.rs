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
    // Rejecting a nested recording must not invalidate a Q8 preparation that
    // the enclosing graph still needs inside a batch (FP16 fallback differs).
    g.graph(106,&mut || {
        g.prequant_gemm_x(&w,n,b.addr(),cols)?;
        assert!(g.graph(107,&mut || panic!("nested graph executed")).is_err());
        g.batch(&mut || run(b.addr(),SAME_X))
    }).unwrap();
    assert_eq!(bits(),b_ref);
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

/// Rejected operations must not mutate an enclosing recording or evict a live graph.
#[test]
#[ignore="requires Vulkan"]
fn rejected_graph_operations_are_inert() {
    let g=Vk::new(0).unwrap();
    let a=common::up(&g,&[0.0f32;4]);
    let b=common::up(&g,&[1.0f32;4]);
    g.graph(201,&mut || g.add(a.addr(),b.addr(),4)).unwrap();
    g.sync().unwrap();
    let mut host=[0xa5u8;16];
    g.graph(202,&mut || {
        assert!(g.graph_drop(201).is_err());
        assert!(g.graph_clear().is_err());
        assert!(g.dtoh(&a,0,&mut host).is_err());
        assert_eq!(host,[0xa5u8;16]);
        assert!(g.htod(&[0xff;16],&a,0).is_err());
        assert!(g.graph(203,&mut || panic!("nested graph executed")).is_err());
        g.batch(&mut || {
            assert!(g.batch(&mut || panic!("nested batch executed")).is_err());
            g.add(a.addr(),b.addr(),4)
        })
    }).unwrap();
    g.graph(201,&mut || panic!("rejected graph_drop evicted graph")).unwrap();
    assert_eq!(common::down::<f32>(&g,&a),vec![3.0;4]);
    g.graph_clear().unwrap();
    let c=common::up(&g,&[1u32,2,3,4,5,6]);
    assert!(g.dtod(&c,0,&c,4,8).is_err());
    g.dtod(&c,0,&c,0,8).unwrap();
    g.dtod(&c,0,&c,16,8).unwrap();
    assert_eq!(common::down::<u32>(&g,&c),vec![1,2,3,4,1,2]);
    eprintln!("R13 FINAL GRAPH PASS: rejected operations inert, device copy disjoint");
}
