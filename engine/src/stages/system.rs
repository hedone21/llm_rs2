//! system Stage 거주지 (§2.1, 확정 거주자 1개).
//!
//! 예정 입주자: `SwitchStage`(hardware 축 동작 — `Arc<Hardware>` 를 register 시점 보관하고
//! `resolve(target)` 으로 대상 backend/memory 를 푼다, §5.1). switch 는 KV migrate 를 *유발*
//! 하지만 주 의도가 device 변경이므로 `system/`. resilience-제어 Stage / observe·보고 Stage 의
//! 분해·명명은 downstream(resilience→stage 매핑 별도 설계).
//!
//! 현재는 골격(입주자 0) — 후속 substep 에서 채운다.
