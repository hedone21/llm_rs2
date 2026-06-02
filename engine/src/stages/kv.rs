//! KV-mutate Stage 거주지 (§2.1).
//!
//! 예정 입주자(Phase α-K): `EvictionStage` / `D2OStage` / `KviQuantizeStage` /
//! `SwapDispatchStage` / `TierMoveStage`(신규). 현 `pressure/{eviction,d2o,quantize,swap}_handler.rs`
//! 의 `handle()` 트리거 부분 + tier_move 신규. 알고리즘(d2o merge ~440 LOC·`offload_one`/
//! `recall_one`)은 `kv/`(현 `pressure/`)로 추출한다(함수 단위 cut, G3-reconcile Q3).
//!
//! `kv/` 정책·포맷에 수평 의존(L3→L3). 현재는 골격(입주자 0) — α-K 진입 시 채운다.
