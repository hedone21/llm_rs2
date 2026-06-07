# Handoff: 3축 plugin-통일 리팩토링 — 호스트 검증 가능 범위 완주 → 남은 건 device 세션

**작성**: 2026-06-07
**HEAD**: `86d398f4 feat(backend): ADR-0005 M-F3 — CPU matmul generic floor`
**브랜치**: master (미푸시)
**작성자**: 메인 세션 (`/goal` 4-마일스톤 완주: MW-D + ADR-0005 M-F1·F2·F3)

**다음 세션 진입 문장**: **"3축 plugin 리팩토링 device 세션 — ADR-0005 step4(KVCacheFormat 해체 + S25 L2 TBT 게이트) + OpenCL backend floor + step5 GpuFold, ADR-0006 Seam B(Phase β decode loop + AB-6). 전부 device(S25/Jetson) 검증 필요라 host `/goal` 범위 밖이었음."**

---

## TL;DR

`/goal` 로 **호스트에서 출력 검증 가능한 남은 plugin-통일 리팩토링 전부를 완주**(4 마일스톤, 4 커밋). Stage 축(ADR-0003/0004)은 기존 완료, 이번에 Weight 축 Seam A 마지막(MW-D)과 Format/Backend 축(ADR-0005) crate 단계 호스트 부분(M-F1·F2·F3)을 닫음. **멈춘 이유**: 남은 작업(step4 L2 device 게이트, OpenCL floor, GpuFold, Seam B)은 전부 S25/Jetson device 검증이 필요해 host evaluator 가 확인 불가 → `/goal` 범위에서 명시적으로 carve-out.

**설계 결정 1건(사용자 adjudication)**: Format/Backend plugin 표면(`KVFormat`/`KVLayoutDesc`/`KV_FORMATS`/`BACKEND_CAPABILITIES`)의 home crate = **technique-api**(ADR D6/L1 준수). goal/handoff 원안의 `engine/format/kv_cache_format.rs` 표기는 ADR 정합으로 정정(engine 은 step4 에서 이 표면을 USE).

---

## 진행 상태 (이번 goal, 4 커밋 — 명시 파일만, 무회귀)

| 마일스톤 | 커밋 | 내용 | 게이트(독립 재검증) |
|---|---|---|---|
| ADR-0006 MW-D | `0ff9f609` | `WeightStageModelCtx` 엔진 `WeightStageCtx` impl (`&TransformerModel` 투영). Seam A 마지막 | stage_ctx 3/3 bit-identical(plan==decider.decide), lib 1246/0 |
| ADR-0005 M-F1 | `7c7906a4` | technique-api `KV_FORMATS`·`BACKEND_CAPABILITIES` 평행 registry + marker trait + find/registered | technique-api 9/9 round-trip |
| ADR-0005 M-F2 | `917af2ff` | `KVLayoutDesc` #[repr(C)] + `ScaleLayout`/`Packing` + `KVFormat::layout()` (L1 repr(C)) | technique-api 9/9 (descriptor read) |
| ADR-0005 M-F3 | `86d398f4` | CPU backend generic floor (`_ => dequant_via_descriptor→matmul_transposed_f32`) + `format/dtype_layout.rs` byte-exact unpacker | dtype_layout 3/3 + common floor 4/4(q8_0 exact, q4_0 특화), lib 1252/0, clippy clean |

전체 게이트: `cargo test -p llm_rs2 --lib`(skip opencl/memory) 1252 passed 0 failed, `cargo clippy --workspace -- -D warnings` 경고 0, 통합 테스트 컴파일 OK.

---

## 다음 작업 (전부 device 세션 — host `/goal` 범위 밖)

1. **ADR-0005 step4 — KVCacheFormat 해체**: compute(`write_kv`/`attention_into`)를 backend dispatch 로, manage(`compact`)만 `KVFormat` 잔류. 검증 = **L2 게이트**(plan-path bit-identical + **S25 TBT Δ≤+3%**, §8). `KVFormat::compact()` 추가는 이 단계. forward_gen_fmt(cold) ↔ plan path(hot) 분기 회귀 주의.
2. **OpenCL backend floor**: M-F3 CPU floor 의 OpenCL 판(descriptor 구동 dequant→f32 matmul). device 검증.
3. **ADR-0005 step5 — GpuFold**: `BackendCapability` 첫 instance(현재 marker trait → 메서드 확정). observe-hook PoC(`/tmp/llm_rs2_poc_hook/`) → `FoldRunner<dyn GpuFold>`.
4. **ADR-0006 Seam B**: `PipelineRegistry` + decode loop 재작성 → `WeightSwapStage`(OneShot) on_phase → `execute_weight_plan`. **AB-6 + Phase β 선행**. 이게 돼야 weight swap 이 production 동작(현재 MW-D ctx + 빌트인은 테스트로만 검증).

---

## Landmines / 미해결

- **호스트 GPU 부재** — 위 4개는 전부 S25/Jetson device 게이트. host 에서 perf/정확성 검증 불가라 `/goal`(host evaluator) 으로 자동화 불가. `run_device.py` 배포 + TBT 측정을 사람이 게이트.
- **L2(최대 함정, ADR-0005)** — fat-LTO 가 crate 단계 cross-crate hot call 을 인라인해 비용을 숨김 → `.so` 가 회귀. "LTO 켜니 빠르다" 신뢰 금지. step4 는 §8 게이트(TBT Δ≤+3%) 필수.
- **M-F3 generic floor 의 raw(F16/F32) Dense 경로는 production 미경유**(특화 arm 이 먼저 잡음) — BF16 만 floor 의 raw 경로를 탄다. dead-ish 이나 family-generic 완결성 위해 보존.
- **`dequant_to_f32_tensor` 가 `b.backend().clone()` 부착** — CPU 무해(f32 slice 만 읽음), OpenCL floor(step4/2) 진입 시 재검토 필요.
- **미커밋 보존**: `engine/Cargo.toml`(microbench_score_readback bin 항목) + `engine/microbench/score_readback.rs` — pre-existing, 건드리지/커밋하지 말 것.
- **dtype 어휘 천장**: `KVLayoutDesc` = block-quant family 만(`TensorDtype` 3-variant 도). mxfp4/codebook/sparse 는 floor 밖 escape.

---

## 참조

- SSOT: `docs/adr/0005-...md`(Format/Backend, D5 floor + D6 registry + L1/L2), `docs/adr/0006-...md`(Weight, D1~D7)
- 메모리: `memory/project_weight_stage_unification.md`(MW-A~D + M-F1~F3 요약)
- 코드 앵커: `crates/technique-api/src/lib.rs`(KV_FORMATS/BACKEND_CAPABILITIES/KVFormat/KVLayoutDesc), `engine/src/format/dtype_layout.rs`(floor unpacker), `engine/src/backend/cpu/{common,x86}.rs`(floor arm), `engine/src/pressure/weights/stage_ctx.rs`(WeightStageModelCtx)
