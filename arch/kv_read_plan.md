# arch: KV read-plan 표면 (ADR-0011 구현 매핑)

> SSOT = `docs/adr/0011-kv-read-plan-surface.md`(표면 D1~D6) + Amendment A1(S2 dispatch).
> 본 문서는 컴포넌트→코드 매핑(HOW). 줄번호 대신 함수/타입 이름으로 참조.

## 1. 컴포넌트 지도

| 컴포넌트 | 위치 | 책임 |
|---|---|---|
| 표면 타입 | `crates/technique-api/src/lib.rs` | KVReadStage / KVReadPlan / ReadGranularity / KVReadStageReg / KV_READ_STAGES / find_read_stage / registered_read_names |
| 빌트인 등록 | `engine/src/kv/read/read_stage_registry.rs`(신규, S5) | Quest 등록 + ensure_builtin_read_stages_registered |
| executor seam | `engine/src/models/transformer.rs::forward_into` / `forward_into_offload` decode arm (S2) | read_plan 호출 + 라우팅 |
| capability | `engine/src/format/selective_read.rs`(trait) + `engine/src/kv/standard_format.rs`(impl) | attention_into_selected, 미지원 = 엔진 폴백 |
| ctx 공급 | `engine/src/kv/eviction/stage_registry.rs::KVStageCtx`(재사용) | tensor(Key)/tensor(QueryStats) |
| page 메타 | Quest stage `&self` Mutex (S4) | K min/max incremental, eviction 후 재구축 |
| CLI | `engine/src/session/cli/kv_mode.rs` (S5) | --read-stage <name> (opt-in, 기본 off) |

## 2. 표면 (S1) — KVCacheStage 거울

KVReadStage 는 KV_CACHE_STAGES/WEIGHT_STAGES 의 4번째 평행 linkme registry.
- pre: ctx 가 유효한 단일-layer StageCtx (immutable borrow, layer attention 직전)
- post: Some(plan) = select 가 ascending 부분집합(granularity 단위) / None = full read(현행 보존)
- INV: read_plan 은 캐시 변형 0 (KVReadPlan 에 new_pos 없음 — ADR D2)

### 구현 현황 (S1 완료)

```rust
// crates/technique-api/src/lib.rs
pub enum ReadGranularity { Token, Page { page_size: u32 } }
pub struct KVReadPlan { pub granularity: ReadGranularity, pub select: Vec<usize> }
pub trait KVReadStage: Send + Sync {
    fn name(&self) -> &str;
    fn read_plan(&self, ctx: &dyn StageCtx) -> Option<KVReadPlan>;
}
pub struct KVReadStageReg { pub name: &'static str, pub make: fn(ReadStageParams) -> Box<dyn KVReadStage> }
#[distributed_slice] pub static KV_READ_STAGES: [KVReadStageReg] = [..];
pub fn find_read_stage(name: &str) -> Option<&'static KVReadStageReg>
pub fn registered_read_names() -> Vec<&'static str>
```

**ADR 표기와 실코드 차이점**:
- ADR D6 에서 `ReadStageParams` / `ReadStageCtx` 가 별도 언급되나, Amendment A1.1d 에서 기존 `StageCtx` 재사용으로 결정. `ReadStageCtx` 신설 없음.
- `ReadStageParams` 는 빌트인 0개 시작이라 현재 `{ _reserved: u8 }` placeholder. Quest 구현(S4) 시 `page_size` 등으로 확장.
- `ReadGranularity` 에 `#[repr(u32)]` 불가 (field 있는 `Page` variant). C-ABI 변환은 `KVReadPlanAbi` 별도 정의로 미래 .so 시점에 위임(ADR §11).

## 3. dispatch 흐름 (S2, Amendment A1) — 미구현(S2 대상)

forward_into decode arm → read_stage = args.read_stage (per-step 1회) → [layer i] if let Some(rs) (분기 1회, INV-147 동형) → KVStageCtx 재사용 → rs.read_plan (dyn call/layer) → Some(plan): fmt SelectiveRead? → attention_into_selected : 미지원 → attention_into (폴백) → None/read_stage None: attention_into (기존 경로 byte-identical).

**seam 위치**: `engine/src/models/transformer.rs::forward_into` decode arm.

## 4. capability 흐름 (S3 완료)

SelectiveRead = opt-in trait. KVCacheFormat base(6-method) 무변경. StandardFormat 첫 구현 (gather 임시 뷰 후 기존 attention 재사용 — 단순 우선). KIVI/opaque/offload 미구현 = 엔진이 plan 무시 + full read.

### SelectiveRead trait

```rust
// engine/src/format/selective_read.rs
pub trait SelectiveRead {
    fn attention_into_selected(
        &self, q: &Tensor, backend: &dyn Backend, out: &mut Tensor,
        dims: AttnDims, select: &[usize], granularity: ReadGranularity,
        scores: Option<&mut [f32]>,
    ) -> Result<()>;
}
```

### StandardFormat 구현 전략 (Tier 1 = 정확성 우선)

1. Page 단위면 → pos 목록으로 전개 (`page_indices_to_positions`).
2. `gather_selected_kv` 로 select 된 토큰을 F32 임시 버퍼(SeqMajor)로 gather.
   - F32: 원소 단위 복사.
   - F16: f16→f32 변환.
   - Q4_0: block dequant → f32 (block 경계 미정렬 pos select 안전 처리).
3. F32 임시 Tensor(SeqMajor, shape `[1, n_sel, kv_heads, head_dim]`)에 `backend.attention_gen` 위임.

**정확성 함정**: softmax 분모는 선택된 부분집합만으로 정규화됨(Quest 의도된 근사). select = 전체이면 bit-identical.

### 테스트 게이트 (3개, standard_format.rs)

- `selective_read_full_select_bit_identical_f32`: select=전체 → attention_into 와 bit-identical (F32/HeadMajor, 오차 <1e-5).
- `selective_read_partial_select_completes_finite`: 절반 select → 에러 없이 완료 + 유한값.
- `selective_read_page_granularity_completes`: Page 단위 select → 완료 + 유한값.

## 5. page 메타 흐름 (S4) — 미구현

Quest stage 가 자기 Mutex 에 page 별 K channel min/max 보유. read_plan 마다 ctx.tensor(Key) 로 incremental. eviction 후 = 다음 read_plan 에서 현 캐시 재반영 (ADR §6, 코어 무수정).

## 6. CLI (S5) — 미구현

--read-stage <name> (KvModeArgs, 기본 미지정=off). 폴백 시 stderr 1회 경고.

## 7. offload 연결 (S6) — 미구현

forward_into_offload decode arm 이 plan.select 를 prefetch 큐 공급원으로 전달. 배선 GREEN only — 8B 부재로 성능 주장 0 (ADR §10, B1=NO).

## 8. Spec Triage = arch-only

신규 INV 0. read-plan 부재=현행 보존, 폴백=정확, 근사 가속(정확성 계약 아님). INV-147/INV-HOTPATH-DISPATCH 의 적용 대상이나 의미 불변.
