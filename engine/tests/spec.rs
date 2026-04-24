#[path = "spec/helpers.rs"]
mod helpers;
#[path = "spec/test_fsm_operating_mode.rs"]
mod test_fsm_operating_mode;
#[path = "spec/test_inv_003.rs"]
mod test_inv_003;
#[path = "spec/test_inv_004_017.rs"]
mod test_inv_004_017;
#[path = "spec/test_inv_005_006.rs"]
mod test_inv_005_006;
#[path = "spec/test_inv_015.rs"]
mod test_inv_015;
#[path = "spec/test_inv_020_026.rs"]
mod test_inv_020_026;
#[path = "spec/test_inv_062_064.rs"]
mod test_inv_062_064;
#[path = "spec/test_inv_072_076.rs"]
mod test_inv_072_076;
#[path = "spec/test_inv_081_082.rs"]
mod test_inv_081_082;

// ── 비-INV Spec 테스트 ──
#[path = "spec/test_eng_alg_010_012.rs"]
mod test_eng_alg_010_012;
#[path = "spec/test_eng_alg_020_022.rs"]
mod test_eng_alg_020_022;
#[path = "spec/test_eng_alg_030_048.rs"]
mod test_eng_alg_030_048;
#[path = "spec/test_eng_alg_060_092.rs"]
mod test_eng_alg_060_092;
#[path = "spec/test_eng_dat_012_031.rs"]
mod test_eng_dat_012_031;
#[path = "spec/test_eng_st_010_035.rs"]
mod test_eng_st_010_035;
#[path = "spec/test_eng_st_052_060.rs"]
mod test_eng_st_052_060;
#[path = "spec/test_proto_010_062.rs"]
mod test_proto_010_062;

// ── Protocol / State / Cross-cutting 테스트 ──
#[path = "spec/test_cross_060_061.rs"]
mod test_cross_060_061;
#[path = "spec/test_eng_st_032.rs"]
mod test_eng_st_032;
#[path = "spec/test_proto_042_073.rs"]
mod test_proto_042_073;

// ── Data / Protocol path 테스트 ──
#[path = "spec/test_eng_dat_c05_streaming.rs"]
mod test_eng_dat_c05_streaming;
#[path = "spec/test_eng_dat_c06_d2o.rs"]
mod test_eng_dat_c06_d2o;

// ── MSG-060 Engine Self-Utilization (2026-04 Phase 1) ──
// MSG-060 필드 17~18, MSG-067~069, INV-091~092
#[path = "spec/test_msg_060_self_util.rs"]
mod test_msg_060_self_util;

// ── Tensor Partition × Plan 통합 테스트 ──
#[path = "spec/test_eng_alg_200_plan_partition.rs"]
mod test_eng_alg_200_plan_partition;
#[path = "spec/test_inv_120_plan_partition_stale.rs"]
mod test_inv_120_plan_partition_stale;

// ── Weight Swap Phase 1 infrastructure (ENG-DAT-092/093/094, ENG-ALG-210) ──
// Runtime dynamic weight swap groundwork. Phase 1 exercises static shape +
// `LayerSlot` atomic primitives; Phase 2 swap execution arrives separately.
#[path = "spec/test_eng_alg_210_initial_load.rs"]
mod test_eng_alg_210_initial_load;
#[path = "spec/test_eng_dat_092_layer_slot.rs"]
mod test_eng_dat_092_layer_slot;
#[path = "spec/test_eng_dat_093_transformer_weights.rs"]
mod test_eng_dat_093_transformer_weights;
#[path = "spec/test_eng_dat_094_secondary_mmap.rs"]
mod test_eng_dat_094_secondary_mmap;
#[path = "spec/test_inv_124_slot_dtype_consistency.rs"]
mod test_inv_124_slot_dtype_consistency;
#[path = "spec/test_inv_125_secondary_mmap_lifetime.rs"]
mod test_inv_125_secondary_mmap_lifetime;

// ── SEQ 통합 테스트 ──
#[path = "spec/test_seq_020_035.rs"]
mod test_seq_020_035;
#[path = "spec/test_seq_040_064.rs"]
mod test_seq_040_064;
#[path = "spec/test_seq_070_093.rs"]
mod test_seq_070_093;
#[path = "spec/test_seq_095_098.rs"]
mod test_seq_095_098;
