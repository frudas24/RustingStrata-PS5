//! Strict integration gate, one immutable environment per process. See tools/ps5/test-r4.sh.
use rusting_strata::{gguf::Gguf, gpu_model::GpuModel, model::Model, tokenizer::Tokenizer, vulkan::Vk};

#[test]
#[ignore = "requires AMD Vulkan and a real generic MoE GGUF"]
fn prefill_scheduler_parity() {
    let path = std::env::var("STRATA_MODEL").expect("set STRATA_MODEL to a generic MoE GGUF");
    let mode = std::env::var("STRATA_PS5_SCHED").unwrap();
    let output = std::env::var("PS5_R4_PARITY_FILE").expect("set PS5_R4_PARITY_FILE");
    let gg = Gguf::open(std::path::Path::new(&path)).unwrap();
    let model = Model::load(&gg).unwrap();
    assert!(model.cfg.is_moe() && model.cfg.hadamard.is_none() && model.loras.is_empty());
    let tok = Tokenizer::from_gguf(&gg).unwrap();
    let prompts = ["A small robot explores a mountain trail and records its discoveries. ",
                   "In Rust, a vector owns its elements and a slice borrows a region of memory. "];
    let inputs: Vec<_> = prompts.iter().map(|p| { let mut ids = tok.encode(&p.repeat(12)).unwrap(); ids.truncate(128); assert!(ids.len() >= 64); ids }).collect();
    let gpu = Vk::new(0).expect("AMD Vulkan GPU required");
    let mut gm = GpuModel::load(&model, Box::new(gpu), 256, 32, 512 << 20, 0).unwrap();
    gm.adapt_every = 0;
    assert_eq!(gm.cache.as_ref().unwrap().resident_count(), 0);
    let mut logits = vec![];
    // Repeat the same prompts to give auto matching shape buckets without cross-process profiles.
    for _ in 0..3 {
        for ids in &inputs {
            gm.reset().unwrap();
            let a = gm.prefill(ids).unwrap();
            assert!(!a.is_empty() && a.iter().all(|x| x.is_finite()));
            // Feed a known identical token regardless of small CPU/GPU rounding differences.
            let b = gm.step(ids[0]).unwrap();
            assert!(b.len() == a.len() && b.iter().all(|x| x.is_finite()));
            logits.extend([a,b]);
        }
    }
    let [trials, cpu_rows, mixed] = gm.ps5_r4_stats();
    eprintln!("R4 integration: {mode} trials={trials} cpu_rows={cpu_rows} mixed={mixed}");
    match mode.as_str() {
        "off" => assert_eq!([trials, cpu_rows, mixed], [0,0,0]),
        "gpu" => assert!(trials > 0 && cpu_rows == 0),
        "half" => assert!(trials > 0 && cpu_rows > 0 && mixed > 0),
        "cpu" | "auto" => assert!(trials > 0 && cpu_rows > 0),
        _ => panic!("unknown R4 mode"),
    }
    let data = serde_json::json!({"mode":mode,"stats":[trials,cpu_rows,mixed],"logits":logits});
    std::fs::write(output, serde_json::to_vec(&data).unwrap()).unwrap();
}
