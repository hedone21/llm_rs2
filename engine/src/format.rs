//! L2 Format 축 추상화 — base trait 거주 모듈 (§4, C2).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §4 / §2.1 C2.
//!
//! format = 축(표현/precision)이다. base trait 정의는 여기(L2 응집 모듈, `capability/` 동급)에
//! 두고, **impl 은 데이터에 내재**한다(KV → `kv/`(현 `pressure/`), weight → `models/weights/`).
//! **guard rail: impl 은 여기 금지** (§2.1).
//!
//! Phase α-W 는 `WeightFormat`(신설, 현 코드 trait 부재)만 입주시킨다. `KVCacheFormat` 은 현
//! `kv_cache_ops.rs` 의 `KVCacheOps` 를 rename·이동하는 **Phase α-K** 에서 `format/kv_cache_format.rs`
//! 로 입주한다(C2 연혁 — `Merge` compact arg 동반). 따라서 α-W 에는 `kv_cache_format` 모듈을
//! 만들지 않는다(빈 모듈 회귀 회피).

pub mod weight_format;

pub use weight_format::{LayerDispatch, SliceSpec, WeightFormat};
