//! Real Vulkan A/B fingerprint. No CUDA requirement, no silent GPU skip.
#![cfg(feature = "ps5-r2")]
mod common;
use rusting_strata::{backend::Backend, vulkan::{Vk, spv_smem::shared_bytes_split}};

#[test]
fn shared_memory_bounds() {
    for wg in [256usize, 512, 1024] {
        for hd in [64, 128, 256] {
            for k in 0..4 { for v in 0..4 {
                let n = shared_bytes_split(&format!("attn_prefill_ps5_{wg}"), &[0, hd], k, v).unwrap() as usize;
                assert!((wg * 32..=wg * 32 + 128).contains(&n), "wg={wg} hd={hd} k={k} v={v}: {n}");
            }}
        }
    }
}

fn vals(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i.wrapping_mul(37).wrapping_add(seed)) % 1009) as f32 - 504.0) / 509.0).collect()
}

#[test]
#[ignore = "physical Vulkan; use tools/ps5/test-r5.sh"]
fn fingerprint() {
    let g = Vk::new(0).expect("Vulkan required");
    let mut bytes = Vec::new();
    // Mixed GQA groups, tail queries, head dims, long position, sparse QSA, fully masked row, sliding window, sink and gate.
    // Keep the 64K synthetic cache small (one KV head) so that the test itself fits even on a constrained PS5.
    for (pos, t, nh, nkv, hd, win, mask, gate) in [
        (0usize, 1usize, 1usize, 1usize, 64usize, 0usize, false, false),
        (29, 37, 12, 4, 64, 0, false, true),
        (63, 17, 20, 2, 128, 31, true, true),
        (127, 9, 33, 1, 256, 129, true, false),
        (255, 15, 8, 2, 128, 1, false, true),
        (65519, 17, 4, 1, 64, 0, false, false),
        (65519, 17, 4, 1, 64, 129, true, true),
    ] {
        let n = (pos + t) * nkv * hd;
        let k = common::up(&g, &vals(n, 11));
        let v = common::up(&g, &vals(n, 97));
        let kc = g.alloc(g.kv_words(n) * size_of::<u16>()).unwrap();
        let vc = g.alloc(g.kv_words_v(n) * size_of::<u16>()).unwrap();
        let zero = common::up(&g, &[0i32]);
        g.kv_store(k.addr(), v.addr(), kc.addr(), vc.addr(), nkv * hd, pos + t, zero.addr(), hd).unwrap();
        let q = common::up(&g, &vals(t * nh * hd, 23));
        let qg = common::up(&g, &vals(t * nh * hd * 2, 71));
        let sink = common::up(&g, &vals(nh, 51));
        let step = common::up(&g, &[pos as i32]);
        let mw = (pos + t).div_ceil(128);
        let bits: Vec<u32> = (0..t).flat_map(|r| (0..mw).map(move |w| if r == 1 { 0 } else { 0x5555_5555u32.rotate_left(((w + r) % 32) as u32) })).collect();
        let qm = common::up(&g, &bits);
        let out = common::up(&g, &vec![f32::NAN; t * nh * hd]);
        g.attn_prefill_m(q.addr(), kc.addr(), vc.addr(), if gate { qg.addr() } else { 0 }, out.addr(), step.addr(), t, nh, nkv, hd,
            1.0 / (hd as f32).sqrt(), if mask { qm.addr() } else { 0 }, mw, if gate { sink.addr() } else { 0 }, win).unwrap();
        let result = common::down::<f32>(&g, &out);
        assert!(result.iter().all(|x| x.is_finite()), "unwritten/nonfinite output at pos={pos} win={win}");
        assert!(result.iter().any(|x| *x != 0.0));
        if mask { assert!(result[nh * hd..2 * nh * hd].iter().all(|x| *x == 0.0)); }
        bytes.extend(result.iter().flat_map(|v| v.to_bits().to_le_bytes()));
    }
    std::fs::write(std::env::var("PS5_R5_PARITY_FILE").expect("set PS5_R5_PARITY_FILE"), bytes).unwrap();
}
