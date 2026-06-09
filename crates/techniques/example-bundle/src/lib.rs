//! 번들 예제 plugin — 한 `.so` 가 **stage 1 + format 1** 을 동시에 export(ADR-0010 E1/E2). 작성자는
//! `register_kv_stage!` + `register_kv_format!` 를 한 crate 에서 호출하고 `export_plugin!()` 1회로
//! 양축 엔트리(register_kv_stages_v2 ⊥ register_kv_formats_v2)를 emit 한다.
//!
//! host dispatcher(`register_dynamic_plugins`)는 이 `.so` 를 1회 dlopen 해 stage-reg·format-reg 를
//! 동일 `Arc<Library>` 공유로 양축 registry 에 분리 등록한다(병합 없음, ADR-0005 D6). "축별 `.so` 분리"
//! 가 불필요함을 실증하는 vehicle(ADR-0010 E7 번들 양축 등록).

use technique_api::{
    KVCachePlan, KVCacheStage, KVFormat, KVLayoutDesc, KeepSpec, Packing, ScaleLayout, StageCtx,
    StageParams,
};

/// 번들 stage — 최근 `target_len` 토큰 유지(example_keep_recent 와 동형, 다른 이름).
struct BundleKeep;
impl KVCacheStage for BundleKeep {
    fn name(&self) -> &str {
        "bundle_keep"
    }
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        let (current, target) = (ctx.current_pos(), ctx.target_len());
        if current <= target {
            return None;
        }
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide((current - target..current).collect()),
            merges: Vec::new(),
        })
    }
}

/// 번들 format — q4_0-like descriptor.
struct BundleFmt;
impl KVFormat for BundleFmt {
    fn name(&self) -> &str {
        "bundle_fmt"
    }
    fn layout(&self) -> KVLayoutDesc {
        KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        }
    }
}

// 한 crate(=한 `.so`)에 양축 — register_kv_stage! + register_kv_format! + export_plugin! 1회.
technique_api::register_kv_stage!("bundle_keep", |_p: StageParams| Box::new(BundleKeep));
technique_api::register_kv_format!("bundle_fmt", || Box::new(BundleFmt));
technique_api::export_plugin!();

#[cfg(test)]
mod tests {
    use super::*;
    use technique_api::{find_kv_format, find_stage};

    #[test]
    fn bundle_registers_both_axes() {
        assert_eq!(find_stage("bundle_keep").expect("stage 등록").name, "bundle_keep");
        assert_eq!(
            find_kv_format("bundle_fmt").expect("format 등록").name,
            "bundle_fmt"
        );
    }
}
