//! Physical Vulkan regression fixture ported from the R6 research archive.
#![cfg(feature = "ps5-r2")]
mod common;
use rusting_strata::{backend::{Backend, ATT_SPLITS}, vulkan::Vk};

#[test]
#[ignore = "requires physical Vulkan; run tools/ps5/test-r7.sh"]
fn masked_decode_has_defined_output() {
    const HD: usize = 64;
    let gpu = Vk::new(0).expect("physical RADV device required");
    let q = common::up(&gpu, &[1.0f32; HD]);
    let k = common::up(&gpu, &[0u16; HD]);
    let v = common::up(&gpu, &[0x3800u16; HD]);
    let step = common::up(&gpu, &[0i32, 1]);
    let sink = common::up(&gpu, &[0.0f32]);
    let part = gpu.alloc(ATT_SPLITS * (HD + 2) * size_of::<f32>()).unwrap();
    let out = gpu.alloc(HD * size_of::<f32>()).unwrap();
    let xq = gpu.alloc(HD + HD / 32 * size_of::<f32>()).unwrap();
    let mut good = true;
    for (count, use_sink, expected) in [(1, false, 0.5), (1, true, 0.25), (0, true, 0.0), (0, false, 0.0)] {
        for use_xq in [false, true] {
            let cells = common::up(&gpu, &[count, 0i32]);
            let input_xq = if use_xq { xq.addr() } else { 0 };
            let sink_addr = if use_sink { sink.addr() } else { 0 };
            let output_xq = gpu.attn_decode_q(q.addr(), k.addr(), v.addr(), 0, out.addr(), part.addr(), step.addr(),
                1, 1, HD, 0.125, i32::MAX, 1, input_xq, cells.addr(), 2, sink_addr, 0).unwrap();
            let values = common::down::<f32>(&gpu, &out);
            let wrong = values.iter().filter(|&&val| !val.is_finite() || (val-expected).abs()>1e-5).count();
            println!("selected={count} sink={use_sink} xq_active={} first={:?} errors={wrong}", output_xq != 0, values[0]);
            good &= wrong == 0;
            if count == 0 && !use_sink && output_xq != 0 {
                let quantized = common::down::<u8>(&gpu, &xq);
                good &= quantized.iter().all(|&byte| byte == 0);
            }
        }
    }
    assert!(good, "decode attention should yield finite, defined values for these cases");
}
