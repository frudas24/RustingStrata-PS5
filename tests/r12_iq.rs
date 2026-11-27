//! Full-output Vulkan parity for short routed integer IQ tiles, compared in separate processes.
#![cfg(feature="ps5-r2")]
mod common;
use rusting_strata::{backend::Backend,quant::GgmlType,vulkan::{Vk,spv_smem::shared_bytes},weights::Tensor};

#[test]
fn shared_bound() {
    for spec in [[0,18,98,0,0], [0,23,136,0,0]] {
        let base=shared_bytes("gemmi_dp",&spec,0).unwrap();
        let small=shared_bytes("gemmi_dp_ps5_iq32",&spec,0).unwrap();
        let medium=shared_bytes("gemmi_dp_ps5_iq64",&spec,0).unwrap();
        assert!(small<medium && medium<base && medium<=32*1024,"LDS {base}/{small}/{medium}");
        eprintln!("R12 LDS type={} base={base} tile32={small} tile64={medium}",spec[1]);
    }
}

#[test]
#[ignore="requires Vulkan"]
fn fingerprint() {
    let g=Vk::new(0).unwrap();
    let mut all=Vec::new(); let mut cases=0;
    for (ty,raw) in [(GgmlType::IQ3XXS,&include_bytes!("fixtures/IQ3_XXS.raw")[..]),(GgmlType::IQ4XS,&include_bytes!("fixtures/IQ4_XS.raw")[..])] {
        assert!(g.int8_gemm(ty));
        for cols in [256usize,512,2048] { for rows in [19usize,64,65] {
            let rb=ty.row_bytes(cols as u64);
            let data:Vec<u8>=raw.iter().copied().cycle().take(rb*rows).collect();
            let w=g.upload(&Tensor{ty,rows,cols,data:&data,scale:1.}).unwrap();
            // Distinct second expert, shift by a whole quantized block.
            let data2:Vec<u8>=raw.iter().copied().cycle().skip(ty.block_bytes()).take(rb*rows).collect();
            let w2=g.upload(&Tensor{ty,rows,cols,data:&data2,scale:1.}).unwrap();
            let xs=cols+32; let ys=rows+3; let n=140usize;
            let host_x=(0..n*xs).map(|i|((i*37%251) as f32-125.)/127.).collect::<Vec<_>>();
            let order=(0..n).map(|i|((i*17+3)%n) as i32).collect::<Vec<_>>();
            let x=common::up(&g,&host_x);
            let ix=common::up(&g,&order);
            // Independent dense oracle with the same input rows gathered on the host.
            let gathered: Vec<f32> = order.iter().flat_map(|&i| host_x[i as usize*xs..(i as usize+1)*xs].iter().copied()).collect();
            let x_indexed=common::up(&g,&gathered);
            for cnt in [1usize,15,16,17,31,32,33,63,64,65,127,128] {
                // Two non-overlapping tiles, different experts and a gap between outputs.
                let second=cnt+4;
                let tile=common::up(&g,&[w.ptr,2u64|((cnt as u64)<<32),w2.ptr,second as u64|(3u64<<32)]);
                for indexed in [false,true] { for act in [0,1,2] {
                    let mut initial=vec![0.125f32;n*ys];
                    if act==0 { for r in (2..2+cnt).chain(second..second+3) { initial[r*ys..r*ys+rows].fill(f32::NAN); } }
                    let y=common::up(&g,&initial);
                    g.gemm(ty,rows,cols,tile.addr(),2,x.addr(),xs,n,if indexed{ix.addr()}else{0},y.addr(),ys,act).unwrap();
                    let v=common::down::<f32>(&g,&y);
                    assert_eq!(v.len(),n*ys);
                    let oracle=common::up(&g,&initial);
                    let xr=if indexed {x_indexed.addr()} else {x.addr()};
                    for (weight,start,len) in [(&w,2,cnt),(&w2,second,3)] {
                        g.gemm_dense(weight,len,xr+(4*start*xs) as u64,xs,oracle.addr()+(4*start*ys) as u64,ys,act).unwrap();
                    }
                    let expected=common::down::<f32>(&g,&oracle);
                    assert_eq!(expected.len(),v.len());
                    assert!(v.iter().zip(&expected).all(|(a,b)| a.to_bits()==b.to_bits()),
                        "routed vs dense: {ty:?} {rows}x{cols} cnt={cnt} indexed={indexed} act={act}");
                    for (r,row) in v.chunks_exact(ys).enumerate() {
                        assert!(row.iter().all(|v|v.is_finite()));
                        assert!(row[rows..].iter().all(|&v|v==0.125));
                        if !(2..2+cnt).contains(&r) && !(second..second+3).contains(&r) {assert!(row.iter().all(|&v|v==0.125));}
                    }
                    assert!(v.chunks_exact(ys).skip(2).take(cnt).any(|r|r[..rows].iter().any(|&v|v!=0.125)));
                    assert!(v.chunks_exact(ys).skip(second).take(3).any(|r|r[..rows].iter().any(|&v|v!=0.125)));
                    all.extend(v.iter().flat_map(|x|x.to_bits().to_le_bytes()));cases+=1;
                }}
            }
        }}
    }
    assert_eq!(cases,1296);
    std::fs::write(std::env::var("PS5_R12_GPU_FILE").unwrap(),&all).unwrap();
    eprintln!("R12 IQ integer GPU parity: {cases} cases {} values",all.len()/4);
}
