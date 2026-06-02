//! `WeightFormat` base trait + dispatch spec (§4.2).
//!
//! KV(`KVCacheFormat`, §4.1)와 대칭이다. base-trait-handle 을 든 Stage 는 dispatch 모드만 알고
//! weight content 는 모른다. precision swap 등 Format mutation 은 concrete-handle Stage
//! (`WeightSwapStage` with `Arc<LayerSlot>`)가 concrete method 로 직접 수행한다.
//! `LayerDispatch::Partition` 의 분산 대상 backend 는 `Hardware`(§3.5)에서 resolve 하고, 슬라이스
//! 마다 다른 (format, hardware) 좌표를 가질 수 있다(GPU-f16 / NPU-q4) — partition = format(표현)
//! × hardware(위치)의 곱.
//!
//! **Phase α-W 신설** — 현 코드 trait 부재. `apply_dispatch` 는 현
//! `transformer.rs:prepare_tensor_partition(ratio, cpu_backend)`(setup 1회, 2-fixed slice)의
//! 일반화다. impl(`PartitionedWeight` → `Vec<WeightSlice>` 일반화)·forward 정적 분기 배선은 후속
//! substep(α-W-5)이며, 본 파일은 trait 표면만 정의한다(아직 소비자 0 — α-W-5 에서 배선).

use anyhow::Result;

use crate::buffer::DType;
use crate::hardware::{DeviceTarget, Hardware};

/// weight layer 의 dispatch 모드(Full / Skip / Partition)를 적용하는 base trait.
pub trait WeightFormat: Send + Sync {
    fn idx(&self) -> usize;

    /// construction tier — companion 을 `Hardware` 로 resolve (§4.2). `Partition` 은 N-HW composite.
    fn apply_dispatch(&self, d: LayerDispatch, hw: &Hardware) -> Result<()>;
    // view() 없음 — read 는 concrete-handle(load_weights) 경유 (§4.2 연혁: 런타임 weight 구조체가
    //               `TransformerLayer` 단 하나 → 제네릭 read view 는 1-adapter 가설적 seam).
    // apply_storage(spec) 없음 — precision swap 등은 concrete-handle Stage(`Arc<LayerSlot>`) 직접 호출.
}

/// construction-time dispatch spec. `Partition` 은 N-slice composite (§4.2, item 2 연혁).
///
/// spec·struct 는 **N-capable 지금** / N-way 병렬 dispatch+merge 커널은 leaf 로 성장
/// (새 HW 추가 시 spec 재형성 0, leaf 한 곳만 — 3축 "M×N 은 Backend 아래 leaf 격리" 원칙).
pub enum LayerDispatch {
    /// 1-slice dense fast-path (slice 기계 우회).
    Full,
    /// 0-slice. Full/Partition 과 나란한 모드로 유지 (stage-축 분리는 friction-triggered).
    Skip,
    /// N-slice composite, share 합 ≈ 1.0.
    Partition(Vec<SliceSpec>),
}

/// per-slice precision (GPU-f16 / NPU-q4) — partition = format(표현) × hardware(위치)의 곱.
pub struct SliceSpec {
    pub share: f32,
    pub hardware: DeviceTarget,
    pub format: DType,
}
