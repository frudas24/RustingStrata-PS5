//! A deterministic cross-process A/B fingerprint. `tools/ps5/test-r2.sh` compares every output bit.
#![cfg(feature = "ps5-r2")]
mod common;
use rusting_strata::{backend::Backend, quant::GgmlType, vulkan::Vk, weights::Tensor};

#[test]
fn small_gemm_shared_memory() {
    use rusting_strata::vulkan::spv_smem::shared_bytes;
    for (id, bytes, old, new) in [(10, 84, 49152, 29184), (12, 144, 40960, 20992)] {
        let spec = [0, id, bytes, 0, 0];
        assert_eq!(shared_bytes("gemmi_dp", &spec, 0).unwrap(), old);
        assert_eq!(shared_bytes("gemmi_dp_ps5_32", &spec, 0).unwrap(), new);
    }
}

#[test]
#[ignore = "physical AMD Vulkan device; use tools/ps5/test-r2.sh"]
fn parity_fingerprint() {
    let g = Vk::new(0).expect("required Vulkan device did not initialize");
    let output = std::env::var("PS5_R2_PARITY_FILE").expect("set PS5_R2_PARITY_FILE");
    let mut bytes = Vec::new();
    let mut append = |v: Vec<f32>| {
        assert!(v.iter().all(|x| x.is_finite()), "non-finite result");
        assert!(v.iter().any(|x| *x != 0.0), "all-zero result");
        bytes.extend(v.iter().flat_map(|f| f.to_bits().to_le_bytes()));
    };
    let cases: [(GgmlType, &[u8]); 5] = [
        (GgmlType::Q2K, include_bytes!("fixtures/Q2_K.raw")),
        (GgmlType::Q4K, include_bytes!("fixtures/Q4_K.raw")),
        (GgmlType::Q8_0, include_bytes!("fixtures/Q8_0.raw")),
        (GgmlType::IQ3XXS, include_bytes!("fixtures/IQ3_XXS.raw")),
        (GgmlType::IQ4XS, include_bytes!("fixtures/IQ4_XS.raw")),
    ];
    for (ty, fixture) in cases {
        let mut widths = vec![256usize, 512];
        if ty == GgmlType::Q8_0 { widths.push(320); }
        for cols in widths {
            let rows = 19usize;
            let rb = ty.row_bytes(cols as u64);
            let data: Vec<u8> = fixture.iter().copied().cycle().take(3 * rows * rb).collect();
            let all = g.upload(&Tensor { ty, rows: 3 * rows, cols, data: &data, scale: 1.0 }).unwrap();
            let w = g.upload(&Tensor { ty, rows, cols, data: &data[..rows * rb], scale: 0.75 }).unwrap();
            let w2 = g.upload(&Tensor { ty, rows: 7, cols, data: &data[rows * rb..(rows + 7) * rb], scale: 1.25 }).unwrap();
            let x_stride = cols + 32;
            let xs: Vec<f32> = (0..65 * x_stride).map(|i| ((i * 37 % 251) as f32 - 125.0) / 127.0).collect();
            let x = common::up(&g, &xs);
            let ids = common::up(&g, &[18i32, 0, 9, 2, 5]);
            for nt in [1usize, 2, 3, 4, 5, 17] {
                let y = g.alloc((nt * rows) * std::mem::size_of::<f32>()).unwrap();
                g.gemvq(&w, x.addr(), x_stride, 0, y.addr(), rows, nt).unwrap();
                append(common::down(&g, &y));
                let a = g.alloc((nt * rows) * std::mem::size_of::<f32>()).unwrap();
                let b = g.alloc((nt * 7) * std::mem::size_of::<f32>()).unwrap();
                g.gemvq2(&w, &w2, x.addr(), x_stride, 0, a.addr(), b.addr(), nt).unwrap();
                append(common::down(&g, &a)); append(common::down(&g, &b));
                let selected = g.alloc((nt * 5) * std::mem::size_of::<f32>()).unwrap();
                g.gemv_rows_n(&w, x.addr(), x_stride, 0, ids.addr(), 5, selected.addr(), nt).unwrap();
                append(common::down(&g, &selected));
            }
            let experts = common::up(&g, &[2i32, -1, 0, 1]);
            for (k, xg) in [(4usize, 4usize), (4, 2), (4, 1)] {
                let y = g.alloc((k * rows) * std::mem::size_of::<f32>()).unwrap();
                g.moe_gemv(ty, rows, cols, all.ptr, rows * rb, experts.addr(), k, x.addr(), x_stride, xg, y.addr(), None, 0).unwrap();
                let out: Vec<f32> = common::down(&g, &y);
                assert!(out[rows..2*rows].iter().all(|&f| f == 0.0));
                append(out);
            }
            if cols % 128 == 0 && g.int8_gemm(ty) {
                for n in [1usize, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
                    // Distinct buffers per case and act; test overwrite, SwiGLU and add semantics.
                    for act in [0, 1, 2] {
                        let y = common::up(&g, &vec![0.125f32; n * rows]);
                        g.gemm_dense(&w, n, x.addr(), x_stride, y.addr(), rows, act).unwrap();
                        append(common::down(&g, &y));
                    }
                }
            }
        }
    }
    std::fs::write(output, &bytes).unwrap();
    eprintln!("R2 parity: {} f32 values", bytes.len() / 4);
}
