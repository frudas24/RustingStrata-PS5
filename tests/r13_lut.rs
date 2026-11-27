//! Verify the actual compiled SPIR-V specialization memory requirements, without a GPU.
#![cfg(feature="ps5-r2")]
use rusting_strata::vulkan::spv_smem::shared_bytes;
#[test]
fn lds_budget() {
    let spec=[0,17,74,0,0];
    let base=shared_bytes("gemm_f_dp_ps5_iq32",&spec,0).unwrap();
    let new=shared_bytes("gemm_f_dp_ps5_iq32_lut",&spec,0).unwrap();
    assert!(new>=base+4096 && new<=65536,"IQ2 LDS {base}/{new}");
    eprintln!("R13 LDS IQ2 {base} {new}");
    for spec in [[0,18,98,0,0],[0,23,136,0,0]] {
        for (base,new) in [("gemmi_dp","gemmi_dp_ps5_lut"),("gemmi_dp_ps5_iq32","gemmi_dp_ps5_iq32_lut"),("gemmi_dp_ps5_iq64","gemmi_dp_ps5_iq64_lut")] {
            let a=shared_bytes(base,&spec,0).unwrap();
            let b=shared_bytes(new,&spec,0).unwrap();
            assert!(b>=a+1024 && b<=65536,"{} {new} LDS {a}/{b}",spec[1]);
            eprintln!("R13 LDS {} {new} {a} {b}",spec[1]);
        }
    }
}
