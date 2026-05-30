# Handoff — #12 attention_into 시그니처 확정 → switch+partition (device-placement) grill 진행 중

**작성**: 2026-05-30
**HEAD**: `3909c783 docs(handoff): Format 용어 정리 종결 → Phase α-W 스레드 재개 진입 문서`
**브랜치**: `master` (코드 0줄 — 본 grill 전부 설계 결정, 미커밋 변경 없음)
**작성자**: 메인 세션 (Claude)
**다음 세션 진입 문장**: **"switch+partition grill 재개 — SW-1 답변부터 (device-placement를 3번째 축으로 분리할지)"**

---

## TL;DR

#12(KVCacheFormat/WeightFormat impl 시그니처 finalize) grill 중 **`attention_into` 시그니처를 완전 확정**했다 (Q1~Q4, backend·scores 파라미터 둘 다 제거). 그 과정에서 사용자가 던진 "tensor partitioning은 Stage냐 Format이냐 + Stage면 backend 받아야 하지 않나"가 새 sub-grill을 열었다 — **switch + partition을 per-step Stage 밖 별도 "device-placement" 축으로 분리할지(SW-1)**. SW-1 질문만 posed, 미답. 멈춘 이유: 사용자가 "새 세션에서 이어서 논의" 요청 (컨텍스트 28%, 자연 분리점).

---

## 확정 결과 (검증 가능) — #12 attention_into

**최종 시그니처** (확정):
```rust
fn attention_into(&self, q: &Tensor, out: &mut Tensor, dims: AttnDims) -> Result<()>;
//  AttnDims { n_heads_q: usize, cache_len: usize }
```

| Q | 결정 | 근거 (코드/결정 인용) |
|---|---|---|
| **Q1** receiver | `&self` + interior mutability | 결정 #8(γ, `Arc<dyn>` 공유). 표준 read 무비용·KIVI scratch만 interior-mut. `Send+Sync`(§4.1)라 RefCell 불가 → GPU handle 공유 + Atomic counter (실현=α-K) |
| **Q2** dims | `AttnDims { n_heads_q, cache_len }` | 이중-출처 차단(cache geometry는 self-derive). cache는 `model.config`로 생성→weight와 단일원본(model_forward.rs:484/548). `kv_start_pos = current_pos - cache_len` self-derive(suffix-window 전제) |
| **Q3** scores | **파라미터 제거** (impl-internal byproduct) | score 이미 multi-dest(GPU accumulator `score_buf` vs `ws.scores`, opencl/mod.rs:440 vs forward_gen.rs:443). param은 1전략만 포착. `layer_idx`도 제거→`self.idx()` self-route(§4.1 geometry). gating flag + read accessor + `needs_attn_scores`/`set_attn_scores` 제거 = **#17 Score domain** |
| **Q4** backend | **파라미터 제거** (Format이 storage로 도달) | `compact`이 이미 `self.k_buffer.backend()` 사용(kv_cache.rs:175/291/381). 별도 held Arc 불요(#11 동반 해소). 리스크 검토: R1 staleness LOW(switch가 `migrate_kv_caches`로 버퍼 재생성, prologue.rs:161)·R2 cycle NONE·R3 Send+Sync NONE·R5 hot-path NONE |

**프레임워크 sharpening** (Q4 리스크 검토에서 도출, SW grill의 전제):
- **same-device 동작** → backend를 소유 storage로 도달 (param/ctx 없음): attention_into / compact / write_kv / partition 실행
- **device-changing 동작** → 타겟 backend가 storage로 도달 불가 → 외부에서 받아야: switch / migrate

---

## 다음 작업 (R5)

### 1순위 — switch+partition grill 재개 (진입점)

**SW-1 (posed, 미답)**: device-placement(switch+partition)를 per-step Stage 밖 별도 session-level **3번째 축**으로 분리할지?

코드 reality (확인 완료):
| | partition | switch |
|---|---|---|
| 트리거 | CLI static (`--tensor-partition`, init.rs:765) | resilience dynamic (`SwitchBackend`→prologue 경계) |
| 셋업 | session init (`prepare_tensor_partition(ratio, cpu_backend)`) | 경계 (`migrate_kv_caches(old,new)`, prologue.rs:161) |
| backend | cpu_backend (init) | old+new (경계) |
| 현재 Stage? | 아님(init+forward) | 아님(prologue) |

- `LayerDispatch` enum **코드 미존재**(§4.2 설계만). `ResilienceAction`에 Partition **없음**(partition=static).
- **추천 = 분리(Framing 1)**: 3축 = Format(noun) / Stage(verb, per-step, same-device, backend 안 받음) / Device-placement(session-level, all-backends). 근거: 코드 정합 + 갈래 B(cross-cutting 자기 패턴) + north star(per-step Stage 불변식 보존).
- **기각 후보 = Framing 2**(OneShot Stage 통합): parent handoff R3가 `OneShotSwitchDeviceStage`/`OneShotPartitionStage`를 Stage 목록에 넣었으나, 일부 Stage만 backend state 갖는 비대칭 재발.
- **분리 후 다룰 nuance**: `LayerDispatch::Skip`(`weight.skip`, executor.rs:648, dynamic·mode-only·backend 불요)은 partition과 lifecycle 다름 → Skip은 가벼운 Stage 가능. **3 variant가 한 집에 안 살 수도.**

### 2순위 — 남은 #12 범위 (attention_into 외)

- **mutation 3** 시그니처: `write_kv`/`write_kv_batch`/`compact` (§4.1 `/* ... */` 채우기). `compact(keep, merges)` atomic만 확정.
- **WeightFormat** `apply_dispatch`/`LayerDispatch`(§4.2) — **switch/partition grill 결과에 의존** (Skip/Partition이 LayerDispatch에 남을지).

### 3순위 — 문서 반영

- arch §4.1 `attention_into` placeholder(`/* q, backend, out, dims, scores */`)를 **확정 시그니처**(`q, out, dims` — backend/scores 제거)로 갱신. **단 switch/partition grill 종결 후 §4.1+§4.2 함께 land 권장**(framework prose 변동 가능성, 중복 편집 회피).

---

## Landmines / 미해결 (R6)

- **코드 0줄** — 전부 설계 결정. 구현은 Phase α-K(Generic→dyn 동행). arch/spec 미반영 상태(위 3순위).
- **§4.1 placeholder가 아직 구식** — `attention_into(&self, /* q, backend, out, dims, scores */)`. 확정형은 backend·scores **없음**. grep으로 §4.1 보면 옛 형태라 혼동 주의.
- **scores/backend 제거는 #17/#11을 *동반 해소* 했지만 실제 구현/method 제거는 별 sprint**: `needs_attn_scores`/`set_attn_scores` 제거 = #17 Score domain. #12는 시그니처만 확정 (forward-compat 보장).
- **SW grill 전제 = "same-device storage 경유 / device-changing 외부 backend"** 원칙. 이게 흔들리면 attention_into Q4도 재검토 (단 attention은 same-device라 영향 없음 — 확정 유지).
- **검증 게이트**: α-K bit-identical(S25 32토큰, 5 paradigm) + avg_tbt Δ≤+3% 가 (a) self.idx() score routing (b) storage-backend 도달 (c) eviction H2O/D2O + KIVI AWQE score 경로 동일성 커버.
- **선행 문서**: 부모 `handoff_kv_weight_grill_2026_05_28.md`(R5 #11/#12) + `handoff_kvcachelayer_attention_into_2026_05_30.md`(④-a, §5.3이 #12 scope 명시). 설계 단일원본 = `arch/pipeline_stage_design_v2.md §4.1/§4.2`.

---

## 자기점검

| 항목 | 확인 |
|---|---|
| 진입 문장으로 첫 명령 가능 | ✓ "switch+partition grill 재개 — SW-1 답변부터" |
| 왜 멈췄는가 | ✓ 사용자 "새 세션에서 이어서 논의" + 자연 분리점(attention_into 확정 직후, SW grill 개시) |
| 최대 landmine | ✓ §4.1 placeholder 구식 + 코드 0줄(설계만) + SW 원칙이 attention_into Q4 전제 |
| 검증 게이트 = 수치/명령 | ✓ α-K bit-identical 5 paradigm + avg_tbt Δ≤+3% |
| 본문 길이 | ✓ 상세 plan은 부모 handoff + v2 §4 링크 위임 |
