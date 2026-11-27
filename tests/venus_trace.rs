//! Real Vulkan diagnostic: no R4 and no output replacement. The shell runner requires exactly one record.
use rusting_strata::{gguf::Gguf,gpu_model::GpuModel,model::Model,tokenizer::Tokenizer,vulkan::Vk};
#[test]
#[ignore = "requires a generic MoE GGUF and Vulkan"]
fn venus_trace_real_moe() {
    let path=std::env::var("STRATA_MODEL").unwrap();
    let gg=Gguf::open(std::path::Path::new(&path)).unwrap();
    let m=Model::load(&gg).unwrap();
    assert!(m.cfg.is_moe() && m.cfg.hadamard.is_none() && m.loras.is_empty());
    let tok=Tokenizer::from_gguf(&gg).unwrap();
    let mut ids=tok.encode(&"A small robot records the result of a Python program. ".repeat(32)).unwrap();
    ids.truncate(128); assert_eq!(ids.len(),128);
    let mut gm=GpuModel::load(&m,Box::new(Vk::new(0).unwrap()),256,32,512<<20,0).unwrap();
    gm.adapt_every=0;
    let a=gm.prefill(&ids).unwrap();
    assert!(!a.is_empty() && a.iter().all(|v|v.is_finite()));
    assert_eq!(gm.ps5_r4_stats(),[0,0,0]);
}
