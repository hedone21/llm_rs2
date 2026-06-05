//! technique-api — 확장 기법(stage 축)이 **엔진 코어 수정 0** 으로 자기를 등록하는 가산 표면.
//!
//! ADR-0003: 확장 메커니즘 = 정적 링크 technique crate + linkme 자동 등록. 각 기법은 별도 crate
//! (`crates/techniques/<name>/`)에서 본 crate 에만 의존해 [`EvictionPlan`] 을 구현하고
//! [`EVICTION_POLICIES`] 슬라이스에 `#[distributed_slice]` 로 자기를 제출한다. 엔진은 construction
//! 시 그 슬라이스를 읽어 정책을 고른다 (closed match arm 제거 → OCP).
//!
//! 의존 방향: `engine → technique-api ← technique crate` (단방향, 순환 없음). 그래서 본 crate 는
//! 엔진 타입(`KVCache`/`Backend`)을 **참조하지 않는다** — stage 표면은 ADR-0003 §D2 의 "planning"
//! 모델이라, 기법은 순수 계산으로 *계획*을 반환하고 버퍼 mutation 은 엔진이 실행한다.

use linkme::distributed_slice;

/// eviction Stage 가 산출하는 병합 지시 — evicted 토큰들(`from`)을 retained 토큰(`into`) 한 자리에
/// 합산한다. 엔진 `format::Merge` 와 동일 의미이며, 가중 없는 균등 병합을 나타낸다 (가중 merge =
/// D2O 는 별도 경로, ADR-0003 §D2·M4).
///
/// `into`/`from` 은 compact 적용 직전(pre-compact) 논리 위치다.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Merge {
    /// 병합 대상 retained 토큰의 위치 (합이 누적될 자리).
    pub into: usize,
    /// 병합될 evicted 토큰들의 위치.
    pub from: Vec<usize>,
}

/// 기법 인스턴스 생성에 필요한 공통 파라미터. 엔진이 CLI args 를 본 struct 로 매핑해 넘긴다
/// (technique-api 가 엔진 args 타입에 의존하지 않도록 평탄한 값만 싣는다).
#[derive(Clone, Copy, Debug)]
pub struct PolicyParams {
    /// sliding window 크기 (최근 유지 토큰 수).
    pub eviction_window: usize,
    /// 앞에서 보호할 prefix 길이 (BOS/시스템 프롬프트 등).
    pub protected_prefix: usize,
    /// heavy-hitter 유지 비율 (H2O 계열).
    pub keep_ratio: f32,
    /// streaming sink(attention sink) 크기.
    pub sink_size: usize,
    /// streaming window 크기 (0 이면 엔진이 기본값 유도).
    pub streaming_window: usize,
}

/// stage 축 **planning 정책** — 순수 계산으로 보존 토큰 *계획*을 산출한다 (ADR-0003 §D2).
///
/// 버퍼를 직접 만지지 않는다: 엔진이 반환된 계획을 `KVCacheFormat::compact(keep, merges)` 로 실행한다.
/// 그래서 본 trait 은 `&mut KVCache` 같은 엔진 타입을 받지 않고 평탄한 좌표/스코어만 받는다 —
/// C-ABI 미래(`cdylib` 승격)와 정합하며, 기법 crate 가 엔진 내부에 결합하지 않게 한다.
pub trait EvictionPlan: Send + Sync {
    /// 정책 이름 (CLI `--eviction-policy <name>` 와 매칭, 로깅용).
    fn name(&self) -> &str;

    /// 보존 토큰 계획 산출 — `keep`(prefix 포함 **ascending**) + 균등 `merges`.
    ///
    /// `importance` Some → score-based(H2O 계열), None → score-free(Sliding 등).
    /// **`None` 반환** = 단일 layer-wide keep-list 로 표현 불가한 정책(예: per-head) — 미지원 신호.
    fn plan_keep(
        &self,
        current_pos: usize,
        target_len: usize,
        importance: Option<&[f32]>,
    ) -> Option<(Vec<usize>, Vec<Merge>)>;
}

/// 한 eviction 기법의 등록 항목. technique crate 가
/// `#[distributed_slice(EVICTION_POLICIES)] static FOO: EvictionPolicyReg = ...` 로 제출한다.
pub struct EvictionPolicyReg {
    /// CLI `--eviction-policy` 이름. 슬라이스 내 유일해야 한다.
    pub name: &'static str,
    /// 파라미터로부터 정책 인스턴스를 만드는 팩토리.
    pub make: fn(PolicyParams) -> Box<dyn EvictionPlan>,
}

/// 전역 등록 슬라이스 — 링크된 모든 technique crate 의 등록이 **링크 타임**에 모인다.
///
/// fat-LTO + `--gc-sections` 가 미참조 섹션을 silent drop 할 수 있다(ADR-0003 §4) — 엔진은 release
/// 빌드에서 기대 정책이 모두 등록됐는지 startup self-test 로 단언해 fail-fast 한다.
#[distributed_slice]
pub static EVICTION_POLICIES: [EvictionPolicyReg] = [..];

/// 이름으로 등록된 기법을 찾는다 (엔진 construction 시 사용).
pub fn find_eviction(name: &str) -> Option<&'static EvictionPolicyReg> {
    EVICTION_POLICIES.iter().find(|r| r.name == name)
}

/// 등록된 모든 기법 이름 (self-test / 진단용).
pub fn registered_names() -> Vec<&'static str> {
    EVICTION_POLICIES.iter().map(|r| r.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dummy;
    impl EvictionPlan for Dummy {
        fn name(&self) -> &str {
            "dummy"
        }
        fn plan_keep(
            &self,
            _current_pos: usize,
            _target_len: usize,
            _importance: Option<&[f32]>,
        ) -> Option<(Vec<usize>, Vec<Merge>)> {
            None
        }
    }

    #[distributed_slice(EVICTION_POLICIES)]
    static DUMMY_REG: EvictionPolicyReg = EvictionPolicyReg {
        name: "dummy",
        make: |_params| Box::new(Dummy),
    };

    #[test]
    fn dummy_registers_into_slice() {
        // linkme 가 같은 crate 의 등록을 슬라이스로 모으는지 확인.
        let reg = find_eviction("dummy").expect("dummy 등록이 슬라이스에 있어야 한다");
        assert_eq!(reg.name, "dummy");
        let params = PolicyParams {
            eviction_window: 8,
            protected_prefix: 4,
            keep_ratio: 0.5,
            sink_size: 4,
            streaming_window: 0,
        };
        let policy = (reg.make)(params);
        assert_eq!(policy.name(), "dummy");
        assert!(policy.plan_keep(10, 8, None).is_none());
    }

    #[test]
    fn registered_names_contains_dummy() {
        assert!(registered_names().contains(&"dummy"));
    }
}
