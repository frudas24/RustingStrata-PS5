//! R7 dense software-dot GEMM candidate: actual GPU output must match the 128-position baseline.
#![cfg(feature = "ps5-r2")]
mod common;
use rusting_strata::{backend::Backend, quant::GgmlType, vulkan::{Vk, spv_smem::shared_bytes}, weights::Tensor};

#[test]
fn shared_memory_bounds() {
    for (id, bytes, old, new) in [(10, 84, 49152, 35840), (12, 144, 40960, 27648)] {
        let spec = [0, id, bytes, 0, 0];
        assert_eq!(shared_bytes("gemmi_dp", &spec, 0).unwrap(), old);
        assert_eq!(shared_bytes("gemmi_dp_ps5_64", &spec, 0).unwrap(), new);
    }
}

#[test]
#[ignore = "physical Vulkan; run tools/ps5/test-r7-gemm.sh"]
fn fingerprint() {
    let g = Vk::new(0).expect("physical Vulkan required");
    let path = std::env::var("PS5_R7_GEMM_FILE").expect("output fingerprint path");
    let cases: [(GgmlType, &[u8]); 5] = [
        (GgmlType::Q2K, include_bytes!("fixtures/Q2_K.raw")),
        (GgmlType::Q4K, include_bytes!("fixtures/Q4_K.raw")),
        (GgmlType::Q8_0, include_bytes!("fixtures/Q8_0.raw")),
        (GgmlType::IQ3XXS, include_bytes!("fixtures/IQ3_XXS.raw")),
        (GgmlType::IQ4XS, include_bytes!("fixtures/IQ4_XS.raw")),
    ];
    let mut out = Vec::new();
    let mut cases_run = 0;
    for (ty, fixture) in cases {
        if !g.int8_gemm(ty) { continue; }
        for cols in [256usize, 512] {
            let xs = cols + 32;
            let x = common::up(&g, &(0..257 * xs).map(|i| ((i * 37 % 251) as f32 - 125.0) / 127.0).collect::<Vec<_>>());
            for rows in [19usize, 64, 65] {
                let rb = ty.row_bytes(cols as u64);
                let data: Vec<u8> = fixture.iter().copied().cycle().take(rows * rb).collect();
                let w = g.upload(&Tensor {ty, rows, cols, data: &data, scale: 0.75}).unwrap();
                let ys = rows + 3;
                for n in [64usize, 65, 127, 128, 129, 255, 256, 257] {
                    for act in [0, 1, 2] {
                        let y = common::up(&g, &vec![0.125f32; n * ys]);
                        g.gemm_dense(&w, n, x.addr(), xs, y.addr(), ys, act).unwrap();
                        let v = common::down::<f32>(&g, &y);
                        for row in v.chunks_exact(ys) {
                            assert!(row[..rows].iter().all(|f| f.is_finite()));
                            assert!(row[rows..].iter().all(|&f| f == 0.125), "padding overwritten");
                        }
                        assert!(v.chunks_exact(ys).any(|r| r[..rows].iter().any(|&f| f != 0.125 && f != 0.0)));
                        out.extend(v.iter().flat_map(|f| f.to_bits().to_le_bytes()));
                        cases_run += 1;
                    }
                }
            }
        }
    }
    assert!(cases_run > 0, "no INT8 GEMM path exercised");
    std::fs::write(path, &out).unwrap();
    eprintln!("R7 GEMM parity: {cases_run} cases, {} f32 values", out.len()/4);
}
