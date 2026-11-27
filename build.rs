// R2 and R5 shaders are optional, built from the same sources as the checked-in R1 SPIR-V.
use std::{env, path::PathBuf, process::Command};
fn main() {
    println!("cargo:rerun-if-changed=src/vulkan/shaders");
    println!("cargo:rerun-if-env-changed=GLSLC");
    println!("cargo:rerun-if-env-changed=SPIRV_VAL");
    if env::var_os("CARGO_FEATURE_PS5_R2").is_none() { return; }
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let compiler = env::var_os("GLSLC").unwrap_or_else(|| "glslc".into());
    let validator = env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into());
    for (name, source, defs) in [
        ("attn_prefill_ps5_256", "attn_prefill_ps5.comp", &["-DSTRATA_R5_WG=256"][..]),
        ("attn_prefill_ps5_512", "attn_prefill_ps5.comp", &["-DSTRATA_R5_WG=512"][..]),
        ("attn_prefill_ps5_1024", "attn_prefill_ps5.comp", &["-DSTRATA_R5_WG=1024"][..]),
        ("gemv_q_ps5_64", "gemv_q.comp", & ["-DSTRATA_GEMV_WG=64"][..]),
        ("gemv_q_ps5_128", "gemv_q.comp", &["-DSTRATA_GEMV_WG=128"][..]),
        ("moe_gemv_q_ps5_64", "moe_gemv_q.comp", &["-DSTRATA_GEMV_WG=64"][..]),
        ("moe_gemv_q_ps5_128", "moe_gemv_q.comp", &["-DSTRATA_GEMV_WG=128"][..]),
        ("gemmi_dp_ps5_32", "gemmi_dp.comp", &["-DSM=1", "-DSTRATA_PS5_SMALL=1"][..]),
    ] {
        let file = out.join(format!("{name}.spv"));
        let status = Command::new(&compiler).current_dir("src/vulkan/shaders")
            .args(["--target-env=vulkan1.3", "-O"]).args(defs).arg("-o").arg(&file).arg(source)
            .status().expect("PS5 R2/R5 requires glslc (or GLSLC=path)");
        assert!(status.success(), "glslc failed for {name}");
        let status = Command::new(&validator).args(["--target-env", "vulkan1.3"]).arg(&file)
            .status().expect("PS5 R2/R5 requires spirv-val (or SPIRV_VAL=path)");
        assert!(status.success(), "spirv-val failed for {name}");
    }
}
