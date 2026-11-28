//! Physical regression for greedy decode at the context boundary (model required).
#![cfg(feature = "ps5-r2")]
use rusting_strata::{gguf::Gguf, gpu_model::GpuModel, model::Model, ops, tokenizer::Tokenizer, vulkan::Vk};

#[test]
#[ignore = "physical Vulkan and explicit GGUF required"]
fn greedy_context_boundary() {
    let path = std::env::var("PS5_R14_CONTEXT_MODEL").expect("explicit GGUF required");
    let g = Gguf::open(std::path::Path::new(&path)).unwrap();
    let m = Model::load(&g).unwrap();
    let t = Tokenizer::from_gguf(&g).unwrap();
    let ids = t.encode("The river town woke before dawn.").unwrap();
    assert!(!ids.is_empty());
    let mut gm = GpuModel::load(&m, Box::new(Vk::new(0).unwrap()), 64, 32, 768 << 20, 0).unwrap();
    let ctx = gm.ctx();
    assert_eq!(ctx, 64);
    let prompt: Vec<u32> = ids.iter().copied().cycle().take(ctx-1).collect();
    let token = ids[0];
    gm.prefill(&prompt).unwrap();
    let want = gm.step(token).unwrap();
    assert_eq!(gm.pos, ctx);
    assert!(want.iter().all(|x| x.is_finite()));
    gm.reset().unwrap();
    gm.prefill(&prompt).unwrap();
    assert_eq!(gm.step_greedy(token).unwrap() as usize, ops::argmax(&want));
    assert_eq!(gm.pos, ctx);
    let err = gm.step_greedy(token).unwrap_err();
    assert!(err.to_string().contains("context full"));
    assert_eq!(gm.pos, ctx);
    gm.reset().unwrap();
    assert_eq!(gm.pos, 0);
    gm.prefill(&prompt).unwrap();
    assert_eq!(gm.step_greedy(token).unwrap() as usize, ops::argmax(&want));
    eprintln!("R14 GREEDY CONTEXT PASS: exact limit, inert rejection, reset and replay");
}
