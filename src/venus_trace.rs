//! Opt-in diagnostic capture of the same expert, same input, CPU shadow vs real GPU prefill.
//! Inspired by UI-Venus causal FFN captures. Does not replace any GPU result.
use crate::{cpu_moe, model::Model, ops};
use serde_json::{Value, json};

#[derive(Clone, Copy)]
pub(crate) struct Target { pub layer: usize, pub pos: usize, pub rank: usize }
impl Target {
    fn parse(value: &str) -> anyhow::Result<Self> {
        let fields = value.split(':').map(str::parse::<usize>).collect::<Result<Vec<_>, _>>()?;
        anyhow::ensure!(fields.len() == 3, "STRATA_PS5_VENUS_TRACE expects layer:absolute_position:router_rank");
        Ok(Self { layer: fields[0], pos: fields[1], rank: fields[2] })
    }
    pub fn from_env() -> anyhow::Result<Option<Self>> {
        match std::env::var("STRATA_PS5_VENUS_TRACE") {
            Ok(v) if v == "off" || v == "0" => Ok(None),
            Ok(v) => Self::parse(&v).map(Some),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

fn compare(a: &[f32], b: &[f32]) -> Value {
    assert_eq!(a.len(), b.len());
    let finite = a.iter().chain(b).all(|v| v.is_finite());
    if !finite || a.is_empty() { return json!({"n":a.len(), "finite":false}); }
    let (mut max_abs, mut base, mut sq, mut base_sq) = (0f64, 0f64, 0f64, 0f64);
    let mut worst = 0;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let (x,y) = (x as f64,y as f64);
        let d = (x-y).abs();
        if d > max_abs { max_abs = d; worst = i; }
        base = base.max(y.abs()); sq += d*d; base_sq += y*y;
    }
    json!({"n":a.len(), "finite":true, "max_abs":max_abs, "worst_index":worst,
        "reference_max_abs":base,"relative_max":if base>0. {Some(max_abs/base)} else {None},
        "rmse":(sq/a.len() as f64).sqrt(),"relative_l2":if base_sq>0. {Some((sq/base_sq).sqrt())} else {None}})
}

/// GPU exposes fused SwiGLU (not separate gate/up). Compare the two stages we actually observe;
/// cross-feed the measured GPU activation into CPU down to isolate downstream arithmetic.
pub(crate) fn capture(m: &Model, target: Target, expert: usize, slot: usize,
    input: Vec<f32>, gpu_act: Vec<f32>, gpu_down: Vec<f32>) -> anyhow::Result<()> {
    let c = &m.cfg;
    anyhow::ensure!(input.len() == c.n_embd && gpu_down.len() == c.n_embd && gpu_act.len() == c.n_ff_exp, "Venus trace shape mismatch");
    anyhow::ensure!(input.iter().chain(&gpu_act).chain(&gpu_down).all(|v| v.is_finite()), "Venus trace non-finite GPU values");
    let g = m.w.layer(target.layer, "ffn_gate_exps.weight")?.expert(expert, c.n_expert);
    let u = m.w.layer(target.layer, "ffn_up_exps.weight")?.expert(expert, c.n_expert);
    let d = m.w.layer(target.layer, "ffn_down_exps.weight")?.expert(expert, c.n_expert);
    let mut gate = vec![0.; c.n_ff_exp]; let mut up = gate.clone();
    cpu_moe::matvec(&g, &input, &mut gate); cpu_moe::matvec(&u, &input, &mut up);
    let act: Vec<_> = gate.iter().zip(&up).map(|(&g,&u)| ops::silu(g)*u).collect();
    let mut down = vec![0.; c.n_embd]; let mut cross = down.clone();
    cpu_moe::matvec(&d, &act, &mut down); cpu_moe::matvec(&d, &gpu_act, &mut cross);
    let grouped = cpu_moe::expert_rows(m, target.layer, std::slice::from_ref(&input), &[(0, expert, 0.)]);
    // FP32 dequant reference, independently of the integer CPU kernels.
    let mut float_gate = vec![0.; c.n_ff_exp]; let mut float_up = float_gate.clone();
    g.matvec(&input, &mut float_gate); u.matvec(&input, &mut float_up);
    let float_act: Vec<_> = float_gate.iter().zip(&float_up).map(|(&g,&u)| ops::silu(g)*u).collect();
    let mut float_cross = vec![0.; c.n_embd]; d.matvec(&gpu_act, &mut float_cross);
    let record = json!({"schema":1,"layer":target.layer,"position":target.pos,"rank":target.rank,"expert":expert,"slot":slot,
        "types":[g.ty.name(),u.ty.name(),d.ty.name()],"scales":[g.scale,u.scale,d.scale],
        "comparisons":{
            "cpu_act_vs_gpu_act":compare(&act,&gpu_act), "cpu_down_vs_gpu_down":compare(&down,&gpu_down),
            "cpu_cross_vs_gpu_down":compare(&cross,&gpu_down), "cpu_grouped_vs_cpu_pair":compare(&grouped,&down),
            "float_act_vs_gpu_act":compare(&float_act,&gpu_act),"float_cross_vs_gpu_down":compare(&float_cross,&gpu_down),
            "cpu_gate_vs_float_gate":compare(&gate,&float_gate),"cpu_up_vs_float_up":compare(&up,&float_up)},
        "values": if std::env::var("STRATA_PS5_VENUS_TRACE_VALUES").as_deref()==Ok("1") {
            json!({"input":input,"gpu_act":gpu_act,"gpu_down":gpu_down,"cpu_gate":gate,"cpu_up":up,"cpu_act":act,
                "cpu_down":down,"cpu_cross":cross,"cpu_grouped":grouped,"float_gate":float_gate,"float_up":float_up,"float_act":float_act,"float_cross":float_cross})
        } else { Value::Null }});
    eprintln!("PS5_VENUS_TRACE {record}");
    anyhow::ensure!(record["comparisons"].as_object().unwrap().values().all(|v| v["finite"] == true), "Venus trace non-finite CPU values");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn venus_trace_metrics() {
        let t = Target::parse("2:42:0").unwrap(); assert_eq!((t.layer,t.pos,t.rank),(2,42,0));
        for bad in ["", "1", "1:2", "1:2:3:4", "-1:2:3", "a:2:3"] { assert!(Target::parse(bad).is_err()); }
        let v = compare(&[0.,3.],&[0.,2.]); assert_eq!(v["max_abs"],1.); assert_eq!(v["relative_max"],0.5);
        assert_eq!(compare(&[f32::NAN],&[0.])["finite"],false);
        assert!(compare(&[1.],&[0.])["relative_max"].is_null());
    }
}
