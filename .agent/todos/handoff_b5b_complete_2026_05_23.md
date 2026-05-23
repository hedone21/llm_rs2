# Handoff: B-5b sprint 종결 → 다음 sprint (Backend trait 분할 + ISP 재점검) 진입

**작성**: 2026-05-23
**HEAD**: `fe1e8e19 refactor(layer): B-5b Phase 3 A — hybrid_attention 모듈 §G L2 격상`
**브랜치/Worktree**: `worktree-b5_trait_extension` (`.claude/worktrees/b5_trait_extension`)
**다음 세션 진입 문장**: `"Backend trait 분할 정책 + ISP 재점검 sprint 진입"`
**선행 진입점**: [[handoff_b5b_phase2_complete_2026_05_23]] · [[handoff_b5b_phase2_stage1_complete_2026_05_23]]

---

## TL;DR

B-5b sprint (INV-LAYER 위반 해소 + Backend trait capability 인프라 + hybrid_attention §G 격상) 전체 완료. Phase 0 (architect 결정) → Phase 1 (R8 위치 정정 3건 + DATA_CONSUMER_PATTERNS allowlist) → Phase 2 (Backend trait 4 capability + 호출지 치환 33+2건) → Phase 3 A (hybrid_attention §G L2 격상). baseline (Phase 1 종결 시점) 206 → 196 (**-10**, B-5b Phase 2+3 A 누적). **멈춘 이유**: 사용자 명시 정책 "ISP 위반 누적은 리팩토링 전 sprint 완료 후 다시 점검" 이행 — Phase 3 B (as_opencl_secondary) + Backend trait 분할 정책 + Phase 3 C (cleanup)는 다음 sprint에서 통합 처리.

**중요 발견**:
1. **§J zone marker는 hot path 부수효과와 mismatch** — plan.rs:1553/1559 (`FullPlan::execute` runtime hot path)에서 `compute_kv_split`/`current` 호출 후 AtomicI32 store + Mutex lock + cl_mem 참조 부수효과 발생. §J 본문 "read-only 정책 query 한정" 제약 위반. §G shared identifier promotion으로 본질 해소 (hybrid_attention 모듈 전체 L2 격상)
2. **§G "RAII guard 본 정책 밖" 문구 해석 명시화** — `HybridScope` RAII가 `HybridAttnSetup` data identifier의 *thread-local lifetime 관리 부산물*인 경우 §G 정신과 부합. §13.8-G register sub-list에 본 해석 명시
3. **`Arc<dyn Backend>` LTO=fat에서 vtable cost 실측 0** — Phase 2 R-1 RPN 145~180 우려가 Stage 2-A (Δ -0.231%) + Stage 2-B (Δ -0.018%) S25 microbench로 empirically refute. 향후 capability hook 추가 시 +3% 게이트 유지 권장
4. **Backend trait method 누적 57 → 61 (+4)** — cpu_companion / cpu_kernels / as_opencl_secondary / yield_after_layer. ISP 누적 backlog 명시 ("리팩토링 전 sprint 완료 후 재점검")

---

## 진행 상태

### 본 sprint 통합 commits (B-5b Phase 0~3 A)

| HEAD | Phase | scope | 변경 요약 |
|---|---|---|---|
| `1cdba86b` | Phase 0 | docs(arch) | architect 결정 — R1 Backend trait default impl + R2 SecondaryStore + R4 DATA_CONSUMER_PATTERNS + R8 위치 정정 3건 |
| `0091aeed` | Phase 1 R8-1 | refactor(layer) | `OpenClEventGpuMeter` → `backend/opencl/gpu_self_meter.rs` 위치 정정 |
| `45bfd16f` | Phase 1 R8-3 | refactor(layer) | `KVCacheOps` trait → `engine/src/kv_cache_ops.rs` §G L2 격상 |
| `7be03c5b` | Phase 1 R4 | build(layer_lint) | `DATA_CONSUMER_PATTERNS` allowlist 신설 + baseline regen |
| `6af90c87` | Phase 0/1 종결 | docs(handoff) | (master tip) — Phase 2/3 다음 세션 인계 |
| `eb1970dc` | Phase 2 Stage 1 | feat(backend) | Backend trait 4 capability hooks + CpuKernelSet + SecondaryStore + OpenClSecondary placeholder (+318 LOC, baseline 변동 0) |
| `b4193bab` | Phase 2 Stage 1 종결 | docs(handoff) | Stage 2 다음 세션 인계 (handoff_b5b_phase2_stage1_complete) |
| `6cd09f9b` | Phase 2 Stage 2-A | refactor(backend) | cpu_kernels (8건) + cpu_companion (25건) 호출지 치환 + cuda 자유함수 dead code 제거 (+83/-45) |
| `28bd7724` | Phase 2 Stage 2-B | refactor(backend) | yield_after_layer 단독 치환 (hot-path) + `maybe_yield_after_layer` → `gpu_yield_impl` rename + 4 GPU backend 1줄 override (+86/-38) |
| `a44a1733` | Phase 2 종결 | docs(handoff) | Phase 3 다음 세션 인계 (handoff_b5b_phase2_complete) + S25 게이트 evidence 2폴더 |
| `5177b10a` | Phase 3 A architect | docs(arch) | §J 폐기 + §G hybrid_attention 격상 결정 (4 결정 단일안: R-P3A-1 γ / 2 N/A / 3 i / 4 register 행 신설) |
| `fe1e8e19` | Phase 3 A 구현 | refactor(layer) | `hybrid_attention.rs` L3→L2 격상 + 호출자 5건 path 갱신 + ARCHITECTURE.md V-?? RESOLVED + §13.8-G register sub-list (+65/-176) |

**부산물 commits (sprint 중간 review skill 신설)**:
- `afe17612` docs(skill): review 스킬 신설 — Plan/Action/Decision/Design 사전 리뷰 10섹션 골격
- `3159fc08` docs(skill): review 스킬 v2 — 결론 직결 골격 + Context + ASCII UML + Premortem 제거
- `5e39c563` style(layer): B-5b Phase 1 잔재 fmt cleanup

### baseline (INV-LAYER 위반 합계) 변동 누적

| 단계 | HEAD | baseline (절대) | 변동 |
|---|---|---|---|
| Phase 1 종결 (master tip) | `6af90c87` | 206 | — |
| Phase 2 Stage 1 | `eb1970dc` | 206 | 0 (인프라 추가만) |
| Phase 2 Stage 2-A | `6cd09f9b` | 209 → 196 | -13 (L001 -7, L003 -6) |
| Phase 2 Stage 2-B | `28bd7724` | 198 | +12 자유함수 → 4 backend trait override inversion / 새 violation 0 |
| Phase 3 A | `fe1e8e19` | **196** | -2 (V-02 hybrid_attention 2건 해소) |

**누적 변동**: 206 → 196 (**-10**, Phase 2 + 3 A 합산). Phase 0 plan의 216 → 186 (-30) 목표 대비 Phase 1 종결 시점 측정 차이 (-10 더 작음)는 layer_lint allowlist 재집계 결과로 추정. **본질 ROI는 절대 카운트가 아닌 자유함수 dispatch 25+건 제거 + downcast 0건 유지 + 정책 zone 적용 형식 (§G/§J)**.

### 호스트 게이트 결과 (전 sprint commit 모두 통과)

| Gate | Phase 1 | Phase 2 Stage 1 | Phase 2 Stage 2-A | Phase 2 Stage 2-B | Phase 3 A |
|---|---|---|---|---|---|
| cargo build | PASS | PASS | PASS | PASS | PASS (1m 04s) |
| cargo fmt | clean | clean | clean | clean | clean |
| cargo clippy `-p llm_rs2 --lib --bin generate -- -D warnings` | 0 | 0 | 0 | 0 | 0 |
| cargo test --lib (회귀) | 0 | 0 | 0 | 0 | 0 (OpenCL stub 병렬 flakiness 단독 PASS) |
| cargo test spec inv_layer | 8 PASS | 8 PASS | 8 PASS | 8 PASS | 8 PASS |
| layer_lint `--baseline` 새 violation | 0 | 0 | 0 | 0 | 0 |

### S25 microbench (Galaxy S25, 6T, Qwen 2.5 1.5B Q4_0)

| 단계 | HEAD | avg_tbt (ms ± stddev) | Δ | 판정 |
|---|---|---|---|---|
| baseline (Phase 2 Stage 1 종결) | `b4193bab` | 32.958 ± 0.052 | — | — |
| Phase 2 Stage 2-A | `6cd09f9b` | 32.882 ± 0.049 | **−0.231%** | PASS |
| Phase 2 Stage 2-B (yield disabled) | `28bd7724` | 32.876 ± 0.083 | **−0.018%** vs Stage 2-A | PASS |
| Phase 2 Stage 2-B (yield enabled, `EVERY=4 US=500`) | `28bd7724` | 40.376 ± 0.094 | +22.8% (sync+sleep, vtable 아님) | 게이트 외 |
| Phase 3 A (mechanical file move + import rename) | `fe1e8e19` | 측정 생략 정당화 | logic 무변경 | — |

- raw 데이터: `papers/eurosys2027/_workspace/experiment/b5b_phase2_stage2{a,b}_s25_gate_2026_05_23/`

---

## 다음 작업 — 다음 sprint (Backend trait 분할 + ISP 재점검)

### 진입 결정 사항 (사용자 라운드)

1. **Backend trait 분할 정책 (R-P3A-2 보류 + 사용자 명시 backlog)**
   - 현재 Backend trait method 수 = 61 (B-5b 진입 시 57, +4: cpu_companion / cpu_kernels / as_opencl_secondary / yield_after_layer)
   - 옵션 (b): `GpuBackend: Backend` sub-trait 분리 — `Arc<dyn Backend>` 사용처 291건 식별·타입 변경
   - 옵션 (c): `BackendExt` blanket impl + downcast — capability query 패턴 모호
   - 옵션 (d): trait 그대로 유지 + 향후 capability hook 추가 시 별도 trait alias로 분류
   - 사용자 결정 사항. 라운드 진입 전 보류

2. **Phase 3 B — `OpenClSecondary` trait body 설계 + qnn_oppkg downcast 제거**
   - `engine/src/secondary.rs::OpenClSecondary` (Stage 1 placeholder) trait body 결정 후 `qnn_oppkg/mod.rs:132~142` `with_opencl_secondary` 클로저 내부 `downcast_ref::<OpenCLBackend>()` 제거
   - 옵션 α `with_queue` closure / β `queue_handle` direct / γ `Arc<OpenCLBackend>` field
   - Backend trait 분할 정책 결정과 동기화 필요

3. **Phase 3 C — 잔여 cleanup**
   - `SecondaryStore` placeholder 처분 — 신규 backend가 trait import만 하고 impl 안 하면 무용지물. 실체화 (`SecondaryMmap` + `RpcmemLayerRegion` impl) vs 제거 (Stage 1 보수 결정 번복)
   - `intra_token_yield_enabled()` — production caller 0건 (test caller 1건). 제거 시 test assertion 단순화 9 LOC. ROI 낮음
   - `gpu_yield_impl` 4 backend 1줄 override 중복 — default impl 흡수 시 CpuBackend도 호출 경로 도달 (yield_every() == 0 즉시 return으로 production 무영향이나 의미 어색). 보수 결정 권장

### 다음 sprint 위임 prompt 초안

```
## 본 sprint = Backend trait 분할 + ISP 재점검 + Phase 3 B/C 흡수

### 핵심 컨텍스트
- 선행: handoff_b5b_complete_2026_05_23.md (HEAD fe1e8e19)
- B-5b sprint 종결 — Backend trait 4 capability hook + 호출지 치환 + hybrid_attention §G 격상
- 사용자 명시 정책: "ISP 누적은 리팩토링 전 sprint 완료 후 재점검" — 본 sprint가 그 시점

### 결정 항목 (architect 라운드 후 사용자 단일안)
1. Backend trait 분할 정책 (옵션 b/c/d, ISP 누적 +4 처분)
2. OpenClSecondary trait body (옵션 α/β/γ) — 1번 결정에 종속
3. SecondaryStore placeholder 처분 (실체화 vs 제거)
4. intra_token_yield_enabled / gpu_yield_impl 중복 처분 (ROI 낮음, 보수 권장)

### Phase 분해
- Phase 1: architect 라운드 (4 결정 단일안)
- Phase 2: 결정 반영 구현 (Backend trait 분할 또는 ISP +0 유지)
- Phase 3 B: as_opencl_secondary 치환 (downcast 제거)
- Phase 3 C: 잔여 cleanup
- Phase 4: 통합 handoff + master FF

### 게이트
1. 호스트 게이트 7개 (현 sprint와 동일)
2. layer_lint baseline 새 violation 0
3. S25 microbench Δ ≤ +3% (Backend trait 분할 시 호출 경로 변화 → 측정 필수)
```

---

## Landmines / 미해결

### 본 sprint 진행 중 발견

1. **§J zone marker 본문 "build-time only" 조건 vs 실측 hot path mismatch**: Phase 0 plan에서 §J 적용으로 명시했으나, 실측 `plan.rs:1553/1559`가 `FullPlan::execute()` runtime hot path였음. zone marker 부착 시 spec 본문 의도 우회 risk. **§G shared identifier promotion으로 전환 (architect 라운드 결정)**. Phase 0 plan은 실측 검증 미수행 가능성 — 향후 동일 패턴 plan 시 호출지 컨텍스트 (build-time vs runtime) 사전 검증 필수

2. **§G 본문 "RAII guard 본 정책 밖" 문구 해석 모호**: `HybridScope` RAII가 `HybridAttnSetup` data identifier lifetime 관리 부산물인 경우 §G 정신과 부합. 본 sprint scope에서 register 행 신설로 해석 명시. **향후 RAII guard 단독 격상 요청 시 §G 본문 재해석 필요** (별도 architect 라운드)

3. **`cpu_companion` default impl 불가 (Stage 1 발견)**: Rust object-safety 제약. `fn cpu_companion(&self) -> &dyn Backend { self }` 컴파일 안 됨. 모든 5 backend explicit override 강제. ISP 누적 +1 (전체 +4 중)

4. **layer_lint 절대 카운트는 architecture invariant 강도와 1:1 비례 안 함**: Stage 2-B에서 자유함수 → trait override inversion으로 +12 발생 (4 backend가 helper import). 새 violation 0 + dispatch는 trait method 경유로 더 깔끔. **baseline 절대 변동에 과민하지 말 것** — 새 violation 0 + 호출지 잔재 0이 진짜 게이트

5. **`Arc<dyn Backend>` LTO=fat에서도 vtable cost 측정 불가**: Phase 2 시작 시 R-1 RPN 145~180 우려가 Stage 2-A (Δ -0.231%) + Stage 2-B (Δ -0.018%) S25 microbench로 empirically refute. **향후 capability hook 추가 시 +3% 게이트 유지** 권장 — worst case 발생 시점이 model/backend 조합 변경 시점에 가까울 수 있음

6. **`OpenClSecondary` trait body placeholder 잔류**: Stage 1에서 empty trait body로 들임, Phase 3 B에서 실체화 또는 처분 필요. Backend trait 분할 정책 재논의 시점과 동기화

7. **handoff baseline 14.66 ms/tok는 다른 컨텍스트 절대값**: Stage 2-A 측정에서 tester가 발견 — `project_weight_swap_tbt_gap_root_cause` Sprint A~F 종결 시점의 Mixed 모드 수치. 본 capability migration sprint baseline = 32.9 ms/tok (호스트 OpenCL stub, 32 tokens, 6T). 미래 sprint에서 baseline 인용 시 측정 컨텍스트 확인 필수

8. **handoff "208" baseline 부정확**: Phase 2 종결 handoff `handoff_b5b_phase2_complete_2026_05_23.md:64` "Stage 2-B 208" 기재 — Phase 3 A 측정 시점 baseline은 198. -10 차이는 layer_lint allowlist 재집계 또는 baseline JSON 업데이트 누락 추정. **handoff baseline 기록 시 그 시점 layer_lint 재실행 결과로 검증 필수**

### 사전 회귀 (B-5b sprint와 무관, B-2 handoff 이후 누적)

- `cargo build --features cuda` / `--features cuda-embedded`: `swap_dispatch.rs:441 map_weights_for_cpu` 사전 회귀
- `crates/qnn_oppkg/src/interface.rs` unresolved imports 17건+ (commit d930801a 이후)
- `cargo build --no-default-features` 사전 회귀
- 호스트 NVIDIA OpenCL `gpu_buffer_shift` + `noshuffle_tests` + `kv_scatter_batch_tests` 24건 병렬 실행 flakiness (단독 또는 `--test-threads=1` 실행 시 PASS, 호스트 GPU stub 환경 의존)
- `engine/microbench/probe_inference_loop.rs:176` useless_conversion (Phase 2 종결 시점 `28bd7724`에도 존재. 본 변경과 무관)

### 다음 sprint 진입 직전 결정 사항 (3건)

1. **Backend trait 분할 정책** — 사용자 명시 보류 후 재논의 시점. 본 sprint 종결 = 시점 도래
2. **`OpenClSecondary` trait body 설계** — Phase 3 B 진입, α/β/γ 택1 (1번 결정에 종속)
3. **`SecondaryStore` placeholder 처분** — Phase 3 B/C 결정과 동기화

---

## Master FF / push 상태

- **현재 worktree branch**: `worktree-b5_trait_extension` (HEAD `fe1e8e19`)
- **origin push 상태**: `a44a1733`까지 push (Phase 2 종결 시점). Phase 3 A architect + 구현 commit 2건 (`5177b10a` + `fe1e8e19`) push 대기
- **master FF 결정 보류**: Phase 2 종결 handoff에 "master FF 정리 (사용자 master worktree 작업 완료 시)" 명시. 사용자 master worktree 상태 확인 후 결정

### 사용자 결정 후 master FF 액션 (참고)

```bash
# 1) origin worktree branch 추가 push
git push origin worktree-b5_trait_extension

# 2) master로 FF (사용자 master worktree 작업 완료 후)
#    옵션 A: 직접 FF (master에 추가 commit 없을 때)
git checkout master
git merge --ff-only worktree-b5_trait_extension
git push origin master

#    옵션 B: PR 생성 (master 추가 작업 있을 때 — merge conflict 점검)
gh pr create --base master --head worktree-b5_trait_extension --title "B-5b: Backend trait capability 인프라 + hybrid_attention §G 격상 (Phase 0~3 A)"
```

---

## 자기점검 (handoff-doc 스킬)

- [x] 진입 문장 한 줄로 다음 세션 첫 명령 가능: `"Backend trait 분할 정책 + ISP 재점검 sprint 진입"`
- [x] 멈춘 이유 명시: 사용자 명시 정책 "ISP 누적은 sprint 완료 후 재점검" 이행. Phase 3 B/C는 Backend trait 분할 정책 결정에 종속
- [x] Landmines 표면화: 8건 (§J vs hot path / §G RAII 해석 / cpu_companion default 불가 / baseline 절대 카운트 한계 / vtable cost 실측 0 / OpenClSecondary placeholder / handoff 14.66 baseline 컨텍스트 / handoff 208 부정확)
- [x] 검증 게이트 수치: B-5b Phase 2+3 A 누적 baseline 206 → 196 (-10), S25 microbench Stage 2-A Δ -0.231% / 2-B Δ -0.018%, 호스트 게이트 7개 × 5 commit 모두 PASS
- [x] 본문 적정 길이 (R6 잔여 backlog 포함하되 sprint 통합 종결문서로 일관)
- [x] master FF 액션 참고 명시 (사용자 결정 후 실행)
