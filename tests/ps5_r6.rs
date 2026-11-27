//! Model-free hardware diagnostics; these are end-to-end transfer measurements, not shader peak bandwidth.
use rusting_strata::{backend::Backend, vulkan::Vk};
use std::time::Instant;

#[test]
#[ignore = "physical AMD Vulkan hardware required; use probe-r6.sh --gpu"]
fn transfer_probe() {
    let gpu = Vk::new(0).expect("AMD Vulkan device required");
    assert!(gpu.ps5_preset());
    let n = 64usize << 20;
    let data: Vec<u8> = (0..n).map(|i| ((i ^ (i >> 12)) & 255) as u8).collect();
    let a = gpu.alloc(n).unwrap();
    let b = gpu.alloc(n).unwrap();
    gpu.htod(&data, &a, 0).unwrap(); gpu.sync().unwrap();
    let t = Instant::now();
    for _ in 0..4 { gpu.htod(&data, &a, 0).unwrap(); gpu.sync().unwrap(); }
    let up = n as f64 * 4.0 / t.elapsed().as_secs_f64() / 1e9;
    let t = Instant::now();
    for _ in 0..8 { gpu.dtod(&a, 0, &b, 0, n).unwrap(); }
    gpu.sync().unwrap();
    let copy = n as f64 * 8.0 / t.elapsed().as_secs_f64() / 1e9;
    let mut out = vec![0u8; n];
    let t = Instant::now();
    for _ in 0..4 { gpu.dtoh(&b, 0, &mut out).unwrap(); }
    let down = n as f64 * 4.0 / t.elapsed().as_secs_f64() / 1e9;
    assert_eq!(data, out);
    let t = Instant::now();
    for _ in 0..128 { gpu.dtod(&a, 0, &b, 0, 4).unwrap(); gpu.sync().unwrap(); }
    let latency = t.elapsed().as_secs_f64() * 1e6 / 128.0;
    assert!([up, copy, down, latency].iter().all(|x| x.is_finite() && *x > 0.0));
    println!("PS5 R6 transfer probe: H2D {up:.3} GB/s D2D payload {copy:.3} GB/s D2H {down:.3} GB/s sync-copy {latency:.3} us");
}
