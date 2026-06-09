//! 예제 format technique crate — "폴더만 추가 = 엔진 코어 수정 0"(ADR-0003) 의 format 축 검증 +
//! 기여자 템플릿. stage 축 [`example-keep-recent`](../example-keep-recent) 의 format 축 짝.
//!
//! 본 crate 는 [`technique_api`] 에만 의존해 [`KVFormat`](technique_api::KVFormat) 을 구현하고
//! `register_kv_format!` 매크로로 자기를 등록한다(정적 linkme + cdylib C-ABI dual-wiring, ADR-0009 D4).
//! descriptor 는 q4_0-like(block_elems 32 / bits 4 / PerBlockF16 / Nibble) — 엔진에 대응 `DType`
//! variant 가 없어 generic floor(opaque)로 처리된다(`synth_q4` 와 동일 부류).
//!
//! GATE-C v2: `cargo build -p example-kv-format --features plugin-cdylib` 로 `.so` 산출 → host 가
//! `register_dynamic_formats` 로 zero-compile dlopen. **엔진이 이 crate 를 force-link 하지 않으므로**
//! (synth_q4 와 달리) 정적 충돌 없이 동적 등록 성공 경로(`make_format` fallback)를 실증하는 vehicle 이다.

use technique_api::{KVFormat, KVLayoutDesc, Packing, ScaleLayout};

/// q4_0-like descriptor 만 제공하는 예제 format(name + layout, 2-method).
struct ExampleKvFormat;

impl KVFormat for ExampleKvFormat {
    fn name(&self) -> &str {
        "example_kv_format"
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

// 등록(dual-wiring, ADR-0009 D4) — 정적: linkme `KV_FORMATS`. 동적(`--features plugin-cdylib`):
// `register_kv_format_v1` C-ABI export. 한 줄로 양쪽.
technique_api::register_kv_format!("example_kv_format", || Box::new(ExampleKvFormat));
// GATE-C v2(ADR-0010 E2): `.so` 엔트리 emit(plugin-cdylib 게이트). 엔진 force-link 안 함 → 동적 등록
// 성공 경로(register_dynamic_plugins → make_format fallback) vehicle.
technique_api::export_plugin!();

#[cfg(test)]
mod tests {
    use super::*;
    use technique_api::find_kv_format;

    #[test]
    fn registers_into_kv_formats() {
        let reg = find_kv_format("example_kv_format")
            .expect("예제 format 등록이 KV_FORMATS 에 있어야 한다");
        assert_eq!(reg.name, "example_kv_format");
        let fmt = (reg.make)();
        assert_eq!(fmt.name(), "example_kv_format");
        let l = fmt.layout();
        assert_eq!(l.block_elems, 32);
        assert_eq!(l.bits, 4);
        assert_eq!(l.scale_layout, ScaleLayout::PerBlockF16);
        assert_eq!(l.packing, Packing::Nibble);
    }
}
