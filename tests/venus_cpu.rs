//! End-to-end CPU text control with fixed teacher tokens and greedy generation in a fresh session.
use rusting_strata::{gguf::Gguf,model::{Model,State},tokenizer::Tokenizer,ops};
#[test]
#[ignore = "requires small local F16 model"]
fn venus_cpu_model() {
    let path=std::env::var("VENUS_F16_MODEL").unwrap();
    let out=std::env::var("VENUS_CPU_OUT").unwrap();
    let gg=Gguf::open(std::path::Path::new(&path)).unwrap();
    let m=Model::load(&gg).unwrap();
    assert!(!m.cfg.is_moe(),"use a small generic dense F16 model");
    let tok=Tokenizer::from_gguf(&gg).unwrap();
    let mut traces=Vec::new(); let mut generations=Vec::new();
    for prompt in ["A small dog went to the park.","def add(a, b):\n    return a + b\n"] {
        let ids=tok.encode(prompt).unwrap(); assert!(!ids.is_empty() && ids.len()<=128);
        let mut st=State::new(&m.cfg); let mut last=Vec::new();
        for id in ids { last=m.step(&mut st,id); assert!(!last.is_empty() && last.iter().all(|v|v.is_finite())); traces.push(last.clone()); }
        let mut gen=Vec::new();
        for _ in 0..8 { let id=ops::argmax(&last) as u32; gen.push(id); last=m.step(&mut st,id); assert!(last.iter().all(|v|v.is_finite())); }
        generations.push(gen);
    }
    std::fs::write(out,serde_json::to_vec(&serde_json::json!({"logits":traces,"greedy":generations})).unwrap()).unwrap();
}
