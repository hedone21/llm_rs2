# Sprint: Backlog Burndown — 잔여 백로그 전체 위임 처리 (2026-06-12)

**위임**: 사용자가 잔여 백로그 전체 처리를 메인 세션(오케스트레이터)에 위임 (2026-06-12).
**구조**: P0 re-triage(PM) → 마스터 플랜 사용자 1회 컨펌 → 트랙 단위 자율 실행.

## 위임 결정 기록 (2026-06-12 사용자)

| 키 | 결정 | 효과 |
|---|---|---|
| A1 | paper/측정 트랙 **제외** | Qwen Microbench Matrix·Llama 재측정·Tier-E = 범위 밖 (사용자 주도) |
| A2 | 5월 성능 잔존군 **제외** | M2.A~J·M3.4·WSWAP-6·mixed precision·long context attention 등 성능 최적화 트랙 = 범위 밖 (거취도 보류 — 사용자 주도) |
| A4 | repo 밖 트랙 **제외** | OSS FC eval = 범위 밖 |
| B1 | 8B/long-context 온보딩 **NO** | 항목 2(K/V 비대칭) 사실상 검증 불가 → 보류 처분 후보. 항목 1/7 재개 조건 미충족 유지 |
| B2 | 멀티세션 1급 지원 **NO** | 로드맵 항목 9 **폐기 처분**. 항목 5(prefix cache)가 해당 수요 커버 |
| B3 | 설계+결정 표기 항목 | 설계안 도출까지 진행 → 결정점만 모아 일괄 질문 |
| C1 | stale 처분 | 재량 처분 + backlog 근거 기록 + 일괄 보고 |
| C2 | 푸시 | 트랙 종결마다 origin 자동 푸시 |
| C3 | 디바이스 | 제약 없음, 게이트 측정은 열누적 쿨다운 규율 준수 |
| C4 | 보고 | 트랙 단위 |
| C5 | 막힌 결정 | 보수적 디폴트(동작 불변) + 기록, 비가역(삭제·spec 의미 변경·성능 회귀 수용)만 질문 |

## P0 re-triage 산출물 (PM 작성)

### (a) 처분 목록 — stale/완료-미표기/범위 밖/보류

> backlog 전수 census(미종결 헤더 본문까지 정독) 후 4분류. backlog Status 갱신은 근거 1줄 동반. line ref = census 시점(2026-06-12) 기준.

#### 분류 1 — 즉시 처분 (backlog Status Edit 완료, 7건)

| line | 항목 | 처분 | 근거 |
|---|---|---|---|
| L38 | host lib 테스트 위생 (γ-3 결정적 버그 2 + POCL-first OpenCL) | **RESOLVED** (헤더만 정리) | 본문이 이미 "RESOLVED 전체 (2026-06-12)" — (a)(b)(c) 전부 해소(`b9775f7d`/`7daa7e69`), 헤더 `[P2-chore]`만 stale |
| L848 | Format 명명 통일 (`*Layer`→`*Format`) | **RESOLVED** | 문서 prose 2026-05-30 전부 완료. 유일 잔여 코드 rename `KVCacheOps`는 α-K BC(2026-06-05)에서 trait 자체 폐기 → rename 대상 심볼 부재 |
| L987 | check_spec_coverage.sh 버그 2건 | **잔여 1건으로 축소** (octal 제거) | octal 해석 버그는 `7daa7e69` base-10 강제로 해소(L38 본문 (c)). 잔여 = INV-DECODE-STAGE ID 추출 부재 1건만 → T1로 |
| L7 | Qwen Microbench Matrix (ACTIVE Sprint) | **사용자 주도 트랙 주석** (Status 미변경) | A1 제외 결정 — burndown 범위 밖. 헤더 주석 + 경고 블록만 추가, 우선순위/Status 미변경 |
| 로드맵 9 | 멀티세션 paging (PagedAttention) | **DROP** | B2=NO — 멀티세션 1급 미지원. 수요는 항목 5(persistence)가 커버. PM 로드맵 질문에 답 도출 |
| 로드맵 2 | K/V 비대칭 merge 가중치 | **보류** (재개=8B 온보딩) | B1=NO — 8B 미온보딩 + 1B score 무가치(항목 0) → AC의 1B WeightedKV ablation 검증 불가 |
| L282 | backend::opencl host test 24 fail | **검증 보류** (해소 가능성 高) | `7daa7e69` GPU-우선 스캔 수정으로 NVIDIA lib 1410/0 보고 → 해소 추정. **단 코드/실행 미검증** — T1에서 실행 1회 후 RESOLVED 처분(단정 금지, 단서만 추가) |

#### 분류 2 — 범위 밖 (A1/A2/A4, 처분하지 않음 — 목록만 기재)

- **paper/측정 트랙 (A1)**: L51 [P2] Llama 3.2 1B 매트릭스 재측정 / L66 [P3] Tier-E 11 op 측정. (L7 Qwen Matrix는 분류 1에서 주석 처리 — 동일 A1 사유.)
- **5월 성능 잔존군 (A2)**:
  - L229 [P0] M3.4 RED pos-baked (QNN architectural blocker, 사용자 architectural decision 대기)
  - L309~361 M2.A~J 11 항목 (QNN-GPU OpPackage M2 layer-graph 전체 — Architect/Implementer/Senior/Tester 혼재)
  - L373~528 WSWAP-6-A/B/C/D/E/F/PREFAULT 7 항목 (weight swap overhead 감축, post-paper 재분류, .cl 커널 + OpenCL async)
  - L286 [P2] Adreno noshuffle GEMV cross-run tuning (Senior, .cl)
  - L574 [P0] Weight Swap layer-level mixed precision & dynamic swap
  - L590 [P0] Long context CPU attention 최적화 (NEON, senior)
  - L790 [P1] Qwen CPU decode gap 해소 (measurement-first, hunch 금지)
- **repo 밖 트랙 (A4)**: OSS FC eval (backlog 본문 부재 — `.agent/todos/feat_oss_fc_model_eval_2026_06_06.md` 별 파일, MEMORY 인덱스 [oss-fc-eval-harness]). 범위 밖.
- **성능 트랙 부속 위생 (범위 밖 분류 권장)**: L278 [P3] qnn_oppkg_poc clippy not_unsafe_ptr_arg_deref 15 errors — QNN 보존 crate(M1 회귀 안전망, read-only) 위생. 성능/QNN 트랙 부속이라 A2 동류로 범위 밖. (T1에 넣지 않음 — read-only crate 정책.)

#### 분류 3 — 트리거/게이트 대기 (개봉 조건 미충족, 그대로 둠)

- **로드맵 1** [보류] Demote op — 게이트 RED 확정(2026-06-12). 재개=8B/long-context 또는 retrieval 실수요.
- **로드맵 6** [P3·트리거] per-head 가변 budget 회계 — 분산 大 부분 충족이나 (a)head-adaptive 실수요 + (b)다층 해상도 확인 2 잔여 미충족.
- **로드맵 7** [P3·트리거] cross-layer cache group — 트리거=8B+ 온보딩(또는 4K+ 상시화) 미충족.
- **로드맵 8** [P3·트리거] windowed attention TensorKind (SnapKV류) — 트리거=항목 3이 prefill-end 압축 못 덮을 때 또는 SnapKV 직접 재현 실수요. 미충족.

> 주: 로드맵 0/3/4는 이미 RESOLVED, 로드맵 4-impl/5는 분류 4(실행 대상)로 이동 — 2026-06-12 분기 취소로 동결 해제됨.

#### 분류 4 — 실행 대상 (작업 2 마스터 플랜으로)

T1~T6 트랙에 배정. 상세는 (b) 참조. 포함 항목(line ref):
- 위생군: L1106 test_inv_layer_005 doc / L1036 arch §13.4/§13.6 / L1048 Precision Swap 다이어그램+qcf_taxonomy / L1088 warmup orphan / L1097 CommandExecutor 2차 census / L979 INV-LAYER 재동결 / L970 test_backend 하네스 / L988(축소) check_spec_coverage 잔여 / L96 §13.8-L 거취 / L248 KiviCache downcast 거취 / L608 다중 모델 검증 매트릭스 / L769 EnergyConstraint / L782 ThermalCollector / L765 QuantizeHandler stub / L241 LISWAP-6 cleanup segfault / L1079 experiments/*.sh argus_eval 이주 / L282(검증 보류) backend::opencl 24 fail
- QCF 신호: L1124 QCF 2-step 핸드셰이크 신호 유실
- KV persistence: 로드맵 5
- read-plan: 로드맵 4-impl (S1~S6)
- 설계+결정: L1135 weight swap 역전 / L698 policy action 반복 방지
- 잔여 P2: L160 generate 분할 잔여 / L1065 argus-eval functional smoke / L82 typed lifecycle hook h-1

### (b) 실행 마스터 플랜 — 트랙 묶음 + 순서 + 게이트

> 묶음 기준: 같은 영역(파일/에이전트/디바이스)·의존성·짧은 것 먼저(모멘텀). 순서 T1→T6. 트랙 종결마다 origin 푸시(C2) + 트랙 보고(C4). 규모 = 트랙 상대 크기 S/M/L. **검증 단정 금지** — 코드 판단 필요한 항목(예: backend::opencl 24-fail 해소)은 트랙 내 "검증 스텝"으로 처리.

---

#### T1 — 위생 일괄 (host-only, 빠름) · 규모 **L**(항목 多, 개당 S)

같은 성격(문서/주석/baseline/거취 결정/stub)이라 한 트랙으로 묶되, 개별 항목은 독립이라 병렬·임의 순서 가능. 디바이스 불요. host 게이트(lib N/0 + fmt + clippy)만.

| # | 항목 (line) | 작업 성격 | 위임 |
|---|---|---|---|
| 1-1 | test_inv_layer_005 doc 헤더 stale (L1106) | 주석 only (`L5_PRODUCTION_BINS` 기준 정정) | Implementer |
| 1-2 | arch §13.4/§13.6 미실현 우산 재작성 (L1036) | 문서 재작성(flat `kv/eviction/` 기준) | Architect |
| 1-3 | Precision Swap 다이어그램 + qcf_taxonomy 위치 정합 (L1048) | 문서 정합(`engine/src/weight/`) | Architect |
| 1-4 | INV-LAYER-001/002/003 baseline 재동결 (L979) | `inv_layer_baseline.json` 위반 동수(8/3/12) 재동결 → 3 spec test PASS | Implementer |
| 1-5 | test_backend 하네스 (MatMulTransposed/Slice/RoPE FAIL) (L970) | reference 산출 또는 tolerance/shape 가정 정정 (production sig MATCH 유지 확인) | Implementer (필요 시 Senior — Q4 block) |
| 1-6 | check_spec_coverage INV-DECODE-STAGE ID 추출 (L988 축소분) | 스크립트 ID 추출 로직 추가 → 신규 갭 0 | Implementer |
| 1-7 | EnergyConstraint spec-code divergence (L769) | spec 갱신 or 수식 연속 변환 — **Architect 판단 필요**(spec 변경이면 Architect 선행) | Architect → Implementer |
| 1-8 | ThermalCollector zone substring 매칭 (L782) | contains 패턴 + 기존 exact 공존 + 테스트 — **필요성 미확정**, 보수적: 현행 유지 권고 후보 | Implementer (or skip) |
| 1-9 | QuantizeHandler stub 제거 + ENG-ALG-092 spec 개정 (L765) | spec ID 걸림 → Architect/spec-manage 선행. struct 삭제 + `target_bits_for_pressure` 강등 + 표 정정 + spec test | Architect → Implementer |
| 1-10 | §13.8-L hot path sub-trait 잔여 7건 거취 (L96) | **결정 항목** — 정적/transitive drag 큼. 보수적 디폴트=status quo 유지(재발동 트리거 명시) → backlog 명시화만 | Architect (거취 1줄) |
| 1-11 | KiviCache hot path downcast 거취 (L248) | §13.8-L과 통합 거취 — 위와 동일 status quo 권고 | Architect (거취 1줄) |
| 1-12 | 다중 모델 검증 매트릭스 (L608) | 디바이스 확보 의존 → **보류 유지** 권고(문서화만, 실행은 디바이스 확보 후) | PM 메모 |
| 1-13 | experiments/*.sh argus_eval 이주 (L1079) | binary 이름 교체(flag 호환) + 실행 검증 | Implementer → Tester |
| 1-14 | LISWAP-6 cleanup segfault (L241) | Drop ordering / cl_mem↔rpcmem_free race. **swap mode 전용·production 무영향** — QNN 보존 trait 영역 인접. 보수적: 분류 재검토(범위 밖 후보) | Architect 거취 판정 |
| 1-15 | backend::opencl 24-fail **검증 스텝** (L282) | `cargo test -p llm_rs2 --lib backend::opencl` 1회 → 0 FAIL이면 RESOLVED 처분 | Tester (실행) → PM(처분) |

- **순서 근거**: T1을 먼저 = 짧고 독립적이라 burndown 모멘텀 + 후속 트랙(특히 spec 변경 동반 T2/T6)의 host 게이트 baseline을 깨끗하게 만든다. 1-10/1-11/1-14는 **거취 결정 항목**(C5: 보수적 디폴트=status quo, 비가역 아니므로 질문 불요 — 1줄 명시).
- **게이트**: 공통 host 게이트(`cargo test -p llm_rs2 --lib` 0 FAIL + `-p llm_manager` 0 FAIL + fmt + clippy `-D warnings`). spec 변경 동반 항목(1-7/1-9)은 Architect 선행 + `cargo test --test spec` + `check_spec_coverage.sh` 신규 갭 0.
- **디바이스**: 불요 (1-15는 host OpenCL=NVIDIA 실행).
- **종결**: origin 푸시 + T1 보고(처분 확정 N건 포함).

---

#### T2 — QCF 2-step 핸드셰이크 신호 유실 (L1124, P2) · 규모 **M**

spec 거동 변경(SEQ-098 timeout) 동반이라 Architect 선행 필수. 디바이스 게이트(S25 prefill-주입 변형) 필요.

- **포함**: L1124 단일.
- **순서 근거**: T1(host 위생) 후 = 깨끗한 spec baseline 위에서 spec 개정. QCF_kv 설계 라운드(2026-06-12 RESOLVED)의 적출 잔여라 맥락 신선.
- **단계**: (1) Architect — `manager/src/lua_policy.rs` `check_qcf_timeout`(:893) timeout 폴백 + `complete_qcf_selection`(:871) late estimate 캐시 설계 + spec SEQ-098 개정 → (2) Implementer — 구현 + `manager/tests/spec/test_seq_095_098.rs` 갱신 → (3) Tester — prefill-중-주입 변형 시나리오 S25 검증.
- **게이트**: spec `cargo test --test spec` + manager test PASS / S25 prefill-주입 변형에서 directive 발화(무-QCF decide, INV-117 cache-miss=0 의미 유지) + verify 매트릭스 30/30 무회귀.
- **디바이스**: 필요 (S25 prefill-주입).
- **위임**: Architect → Implementer → Tester.

---

#### T3 — 세션 KV persistence 항목 5 (Tier 1, P3·1B 체감 효익) · 규모 **M~L**

C 군집 중 1B 현 타겟에서 TTFT 즉시 체감 나오는 유일 항목. 분기 취소로 동결 해제.

- **포함**: 로드맵 5 (Tier 1만).
- **순서 근거**: 항목 1–4와 독립(IR/TensorKind 무접촉)이라 T4와 병렬 가능하나, format snapshot capability 신설이 read-plan(T4)보다 단순+체감 큼 → 먼저. (B2=NO로 멀티세션 수요를 본 항목이 흡수하는 책임도 부여됨.)
- **단계**: (1) Architect — format snapshot capability(`snapshot(range)→bytes`/`restore(bytes,at_pos)`) capability-handle 설계 + 스냅샷 헤더(model_hash/format_id/tokenizer_hash/token_ids) + 세션 API(`save_prefix`/`try_restore_prefix`) → (2) Implementer — 구현(F32/F16/Q4_0) + 무효화 3케이스 → (3) Tester — S25 동일 prompt 2회차 TTFT 단축 실측 + 복원 후 greedy token-id == fresh prefill 일치.
- **게이트**: 복원 후 greedy token-id byte-identical(fresh prefill 대비) + 무효화 3케이스(model/format/tokenizer 변경) 테스트 + S25 2회차 TTFT 단축 실측 + α-K frozen 무회귀.
- **디바이스**: 필요 (S25 TTFT 게이트).
- **위임**: Architect → Implementer → Tester.

---

#### T4 — read-plan 4-impl (S1~S6, ADR-0011) · 규모 **L**

ADR-0011이 표면 SSOT. 단계별 진행. 분기 취소로 동결 해제(머지 후 재-triage 게이트 소멸 — 단 ADR §8 #3 의미 계약 보존).

- **포함**: 로드맵 4-impl (S1~S6).
- **순서 근거**: 항목 3(QueryStats, ✅ RESOLVED) 신호 공급원 충족. T3 후 = persistence의 snapshot capability 패턴(capability-handle)이 S3 format capability(`SelectiveRead`)의 선례가 됨. **단 hot-path seam(S2)이 위험** → ADR R1(RPN 378) dispatch 전략 별도 amendment 고정 필요.
- **단계 (ADR D1~D5)**: S1 technique-api 표면(trait+plan+registry) → S2 엔진 executor seam(★hot-path, read stage 부재 시 분기 1회, dispatch 전략 amendment) → S3 format capability(Standard format 첫 구현, 미지원=full read 폴백) → S4 page 메타 유지(Mutex K min/max + QueryStats incremental) → S5 첫 빌트인 Quest(CLI `--read-stage quest` opt-in) → **S6 offload prefetch 정합 — 8B 없음(B1=NO)이라 offload prefetch 연결만 하고 측정 주장 없음 명기**(ADR §10 — 1B 성능 주장 금지).
- **게이트**: α-K frozen byte-identical(read stage 부재 happy path) + 폴백 bit-identical(미지원 format) + read 활성 시 PPL/EMR 근사 일치(정확 일치 아님) + TBT Δ≤+3%(ADR-0005 §8) + lib N/0 + clippy clean + 기존 .so dlopen 호환. **S6는 측정 주장 없이 배선 GREEN만.**
- **디바이스**: 필요 (S2 TBT 게이트, S5 품질 메트릭).
- **위임**: Architect(S1 표면 + S2 dispatch amendment) → Senior Implementer(S2 hot-path seam) → Implementer(S1/S3/S4/S6) → Tester(게이트).

---

#### T5 — 설계+결정 항목 (B3) · 규모 **M** (설계 위주)

설계안 도출까지 진행 → 결정점만 모아 (c)에서 일괄 질문. 구현은 결정 후.

- **포함**: L1135 weight swap 역전(RestoreDefaults F16 recall) + L698 policy action 계열 반복 방지.
- **순서 근거**: 둘 다 "설계 라운드 선행 + 사용자 결정" 표기 항목이라 묶어서 결정점을 (c)로 모은다. 비가역(메커니즘 신설·정책 거동 변경)이라 C5 질문 대상.
- **단계**: Architect 설계안 도출 → 결정점 (c) 예고 → 사용자 일괄 답 → Implementer 구현 → (weight swap 역전은 S25 device 게이트).
- **게이트**: 결정 후 — weight swap 역전: RestoreDefaults 수신 시 F16 복원 + 복원 후 출력 == swap-전 happy baseline(S25). action 반복 방지: 선택 방식별 단위 테스트 + 시뮬레이션 재현(2026-04-15 사례 비순환).
- **디바이스**: weight swap 역전만 필요 (S25).
- **위임**: Architect(설계) → 사용자(결정) → Implementer → Tester.

---

#### T6 — 잔여 P2 · 규모 **M**

- **포함**: L160 generate 분할 잔여(argus-chat bin화 + Manager IPC 통합) / L1065 argus-eval functional smoke(Linux/S25 런타임) / L82 typed lifecycle hook h-1.
- **순서 근거**: 마지막 = argus 패밀리 분할은 사실상 완성(γ-3)이라 잔여 소량 + typed hook h-1은 가설 단계라 설계 선행. **L82(h-1)는 T5와 병합 검토** — inference loop 재설계 + design round(Architect+사용자) 필요라 B3 결정 묶음에 자연(아래 (c) 참조). L1065는 환경(Linux/S25 OpenCL)만 확보되면 즉시 가능한 Tester 단독.
- **단계**: (argus-chat) Architect 설계 → Implementer bin화 + Manager IPC / (smoke) Tester E2E 5모드 / (h-1) Architect design round(T5 병합 시 (c) 결정에 포함).
- **게이트**: argus-chat bin bit-identical(legacy 동일 모드) + Manager IPC mock 시나리오 / argus-eval 5모드 E2E PASS + `cargo test -p llm_rs2` 전체 PASS + spec suite PASS.
- **디바이스**: argus-eval smoke = Linux 또는 S25 / argus-chat = S25 bit-identical 게이트.
- **위임**: Architect → Implementer → Tester.

---

#### 트랙 의존 그래프 (요약)

```
T1 (위생, host) ──┬─→ T2 (QCF 핸드셰이크, spec+device)
                  ├─→ T3 (persistence, device) ──→ T4 (read-plan, device) [snapshot capability 선례]
                  ├─→ T5 (설계+결정, B3) ──→ (c) 일괄 질문 ──→ T5 구현
                  └─→ T6 (잔여 P2) [h-1은 T5 결정 묶음 검토]
```

- T1은 모든 후속의 host baseline. T2/T3/T5/T6은 T1 후 상호 병렬 가능(자원 1개면 순차 T2→T3→T4→T5→T6).
- T4는 T3 후 권장(capability-handle 패턴 선례) + 항목 3 충족(✅).
- **트랙 종결마다**: origin 푸시(C2) + 트랙 보고(C4). 비가역 결정(C5)은 T5에서 (c)로 사전 질문.

### (c) 결정점 예고 — B3 일괄 질문 대상

> B3 = "설계안 도출까지 진행 → 결정점만 모아 일괄 질문". 아래는 T5(+T6 h-1) 설계 라운드에서 사용자에게 돌아갈 예상 결정. Architect 설계안이 옵션을 구체화하면 이 목록 기준으로 일괄 질문. C5(비가역만 질문) 기준 — 메커니즘 신설·정책 거동 변경·spec 의미 변경은 질문, 동작 불변 디폴트는 자율.

#### D-1. weight swap 역전 (T5, L1135) — recall 트리거 정책

- **결정 본질**: RestoreDefaults 수신 시 swap된 layer를 F16으로 복원하는 **신규 메커니즘**(F16 recall / dual-resident + Consumed된 OneShot 역방향 재기동)을 도입할지, 도입한다면 트리거 조건.
- **예상 옵션**:
  - (A) RestoreDefaults에서 무조건 F16 recall — 압력 해소 즉시 품질 복원. 메모리 이득 반납.
  - (B) 품질 회복 시나리오 명시 트리거만 recall — 평시 Q4 유지(production winner Incremental의 메모리 이득 보존), 특정 신호(예: 정확도 회복 directive)에서만.
  - (C) 현행 유지(no-op) — 역전 미구현, 본 항목 보류 유지.
- **비가역 포인트**: 메커니즘 신설(dual-resident 메모리 모델 + report `from/to_dtype` 하드코딩 해제) + partition+precision 합성 역전(ADR-0006 §6) 처분. **질문 필요.**
- **보수적 디폴트(질문 전 가정)**: (C) 현행 no-op 유지 — 동작 불변.

#### D-2. policy action 계열 반복 방지 (T5, L698) — 방식 선택

- **결정 본질**: external injection 지속 시 action 순환(kv_quant↔kv_evict 교체 반복) 방지 방식. 정책 거동 변경이라 결정 필요.
- **예상 옵션** (backlog 본문 A~D):
  - (A) cooldown — observation window(3s) 외 "최근 N관측 모두 낮으면 동일 **계열** 재선택 쿨다운"(ctx.history 활용).
  - (B) 계열(category) 개념 — `kv_evict_*`/`kv_quant_*` 묶어 계열 내 교체 억제.
  - (C) 외부 압박 감지 — memory slope 상승 중이면 relief 관측 불신(ctx.history slope).
  - (D) 현행 유지 — "낮은 relief action 자연 교체가 올바른 동작" + production엔 injection 없으니 real signal 정확. **backlog 주의: D가 맞을 수도 있음.**
- **비가역 포인트**: policy_default.lua / lua_policy.rs 거동 변경. 단 production injection 부재 가정이 맞으면 변경 자체가 불필요. **질문 필요**(특히 D 채택 여부 = 변경 안 함 결정).
- **보수적 디폴트(질문 전 가정)**: (D) 현행 유지 — 실기(S25) 테스트 전 변경 회피(backlog 권고 일치).

#### D-3. typed lifecycle hook h-1 (T6, L82 — T5 결정 묶음 검토) — 착수 여부

- **결정 본질**: inference loop 전체 재설계(Phase 4-4 후속, 1~2주 범위)를 본 burndown에서 착수할지, 별 sprint로 분리할지. design round(Architect+사용자) 필수 표기 항목.
- **예상 옵션**:
  - (A) 본 burndown에서 설계 라운드 진입 — PressureHook/KvCacheHook/PrefetchHook typed 격리.
  - (B) 별 sprint 분리 — 범위(1~2주) + inference loop 재설계 위험이 burndown 성격(잔여 처리)과 mismatch.
- **비가역 포인트**: 착수 자체는 가역(설계 단계). 단 범위가 burndown 트랙 1개보다 큼. **질문 필요**(burndown 범위에 포함할지).
- **보수적 디폴트(질문 전 가정)**: (B) 별 sprint 분리 — burndown은 L82를 "설계 라운드 예약"으로만 표기, 실착수는 사용자 컨펌 후.

#### 그 외 트랙의 비가역 결정 예상 지점 (C5)

- **T1-1-9 (QuantizeHandler stub 제거)**: ENG-ALG-092 spec 표 정정 = **spec 의미 변경**(제목 "6종" ↔ 표 4개 stale + "완료"↔never-registered NoOp divergence). spec 거동이 아니라 문서-구현 정합이라 비가역 약함 — Architect 판정으로 처리, divergence 해소 방향(struct 삭제 vs free fn 강등)이 갈리면 (c)에 추가.
- **T1-1-7 (EnergyConstraint divergence)**: "spec을 코드에 맞게 갱신" vs "코드를 spec 수식대로 연속 변환" = **spec 의미 변경 가능**. backlog Notes "spec 갱신이 더 현실적". 기능 동작은 현행 동일하므로 spec 갱신(코드 불변) 방향이면 보수적 — 코드 변경(이산→연속) 선택 시 directive 거동 변동이라 질문. 보수적 디폴트=spec 갱신(코드 불변).
- **T1-1-10/1-11/1-14 (§13.8-L / KiviCache downcast / LISWAP-6 거취)**: 전부 **status quo 유지가 보수적 디폴트**(동작 불변) — 비가역 아니므로 질문 불요, backlog에 재발동 트리거 1줄 명시로 종결. LISWAP-6은 범위 밖 재분류 가능성(Architect 거취 판정).
- **T4-S2 (read-plan hot-path dispatch 전략)**: 정적 vs dyn dispatch = ADR R1(RPN 378). **별도 amendment로 고정**(ADR가 SSOT)이라 사용자 질문보다 Architect amendment. 단 TBT Δ>+3% 회귀가 측정되면(성능 회귀 수용 = 비가역) 질문.
- **T2 (QCF 핸드셰이크 SEQ-098 개정)**: spec timeout 거동 변경 = **spec 의미 변경**이나 "신호 유실 복원"이라 방향 단일(timeout 시 무-QCF decide). Architect 선행으로 처리, 방향 분기 없으면 질문 불요.

> **일괄 질문 시점**: T5 진입(Architect 설계안 도출 직후)에 D-1/D-2/D-3 + 위 조건부 항목을 한 번에. 그 전 트랙(T1~T4)은 보수적 디폴트로 자율 진행, 비가역 지점 도달 시에만 개별 에스컬레이션.

## 공통 게이트 (전 트랙)

- host: `cargo test -p llm_rs2 --lib` 0 FAIL (1410 기준) + `cargo test -p llm_manager` 0 FAIL + fmt + clippy `-D warnings`
- device: α-K frozen 3-dtype 결정론 라인 byte-identical + tbt Δ≤+3% (쿨다운 후 측정) / S25 verify 매트릭스 **30/30** 무회귀
- spec 변경 동반 시: Architect 선행 + `cargo test --test spec` + `scripts/check_spec_coverage.sh` 신규 갭 0
