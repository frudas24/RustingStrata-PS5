//! Physical Vulkan regression fixture ported from the R6 research archive.
#![cfg(feature = "ps5-r2")]
mod common;

use rusting_strata::{backend::Backend, vulkan::Vk};

#[test]
#[ignore = "requires physical Vulkan; run tools/ps5/test-r7.sh"]
fn publishes_activations_when_any_route_misses() {
    const N: usize = 32;
    const TOP: usize = 4;
    const TOKENS: usize = 8;
    const HIDDEN: usize = 64;
    const ROUTED: usize = TOP * TOKENS;
    const ACTIVATIONS: usize = 128;

    let gpu = Vk::new(0).unwrap();
    let hidden: Vec<f32> = (0..TOKENS * HIDDEN).map(|n| (n+1) as f32).collect();
    let h = common::up(&gpu,&hidden);
    let ids = gpu.alloc(ROUTED * 4).unwrap();
    let weights = gpu.alloc(ROUTED * 4).unwrap();
    let slots = gpu.alloc(ROUTED * 4).unwrap();
    let weighted_hit = gpu.alloc(ROUTED * 4).unwrap();
    let (mailbox, host) = gpu.host_mailbox(ACTIVATIONS + hidden.len()).unwrap();

    for (misses, distributed) in [(0,false),(1,false),(2,true),(ROUTED,true)] {
        let mut logit_data = vec![-100.0f32; N*TOKENS];
        let mut expert_data: Vec<i32> = (0..N as i32).collect();
        if distributed {
            for expert in 0..misses {
                expert_data[(expert % TOKENS)*TOP + expert/TOKENS] = -1;
            }
        } else if misses==1 {
            expert_data[0] = -1;
        }
        for token in 0..TOKENS {
            for lane in 0..TOP {
                logit_data[token*N + token*TOP + lane] = (TOP-lane) as f32;
            }
        }
        let p = common::up(&gpu,&logit_data);
        // Compare against selected ids, with multiple tokens in separate routing subgroups.
        let expected: Vec<i32> = (0..ROUTED as i32).collect();
        let expected_slots: Vec<i32> = expected.iter().map(|&id| expert_data[id as usize]).collect();
        let table = common::up(&gpu,&expert_data);
        for sequence in 1..=20u32 {
            gpu.memset(&mailbox).unwrap();
            let step = common::up(&gpu,&[0u32,sequence]);
            gpu.moe_route(p.addr(),N,TOP,1.0,table.addr(),ids.addr(),weights.addr(),slots.addr(),
                weighted_hit.addr(),h.addr(),mailbox.addr(),HIDDEN,ACTIVATIONS,step.addr(),TOKENS,0,false).unwrap();
            gpu.sync().unwrap();
            let read = |offset| unsafe{host.add(offset).read_volatile()};
            assert_eq!(read(0),sequence);
            assert_eq!(read(1),TOKENS as u32);
            assert_eq!(read(8),0);
            let actual = common::down::<f32>(&gpu,&weights);
            for lane in 0..ROUTED {
                assert_eq!(read(32+lane),expected[lane] as u32);
                assert_eq!(read(32+ROUTED+lane),actual[lane].to_bits());
                assert_eq!(read(32+2*ROUTED+lane),expected_slots[lane] as u32);
            }
            for (i,val) in hidden.iter().enumerate() {
                assert_eq!(read(ACTIVATIONS+i),if misses>0 {val.to_bits()} else {0});
            }
        }
        println!("misses={misses},repetitions=20,passed=true");
    }
}
