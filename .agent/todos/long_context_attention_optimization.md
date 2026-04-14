# Long Context CPU Attention 최적화 — 4K+ 성능 갭 해소

(과거 섹션 1~10은 이전 커밋 참고; §11~§12만 최신 상태)

## 12. 다음 세션 시작 가이드 (2026-04-14 21:20 — 3차 세션, **KV stride 가설도 기각**)

### 🆕 2026-04-14 3차 세션: standalone microbench로 KV stride 가설 격리 검증 → **REJECT**

**목적**: 심야 세션의 "host KV stride 차이가 Adreno 메모리 계층 동작 차이를 만든다" 가설을 production 노이즈 없이 격리 측정.

**방법**: `engine/src/bin/microbench_flash_attn.rs` 신규. production `flash_attn_f32_f16.cl` 무수정 컴파일, Q1 커널 단독 dispatch. 두 KV layout으로 같은 데이터를 다른 stride로 읽도록 strides arg만 변경.

| layout | k_nb1 | k_nb2 | run1 slope | run2 slope |
|---|---:|---:|---:|---:|
| HeadMajor (현 production) | 256 B | cap·dk·2 | 0.3635 | 0.3903 |
| PosMajor (llama-view 등가) | 512 B | dk·2 | 0.3652 | 0.3836 |

- **기울기 비 0.995× / 1.018×** (run 1 / run 2) — 노이즈 범위 < 2%
- **KV stride 가설 기각** — Adreno 830에서 두 access pattern 사실상 동등

**부가 발견 (참고용)**: microbench attention slope **0.36~0.39 μs/n_kv** = production attention slope (13.23 μs/n_kv) ÷ 28 layers = 0.473 μs/n_kv per layer. 1.3× 차이는 thermal/cache pressure/mask/Q variance로 설명 가능. 즉 production 측정값 자체는 합리적이고, layout이 "느림"의 원인은 아님.

**산출물**: `.agent/research/microbench_flash_attn/run1_2120.txt`

### 🆕 2026-04-14 21:30: Option C — Per-op slope vs TOTAL 모순 검증 → **모순 아님**

**원본**: `.agent/research/option_2b_raw/llm_rs2_ctx*.log` (decode per-op breakdown), `llama_ctx*.csv` (per-kernel)

**같은 profile-events 세션 내 슬롭 합 계산**:

| op | slope μs/n_kv | % of profile sum |
|---|---:|---:|
| attention      | 13.23 | 93.6% |
| matmul_ffn     | 0.393 | 2.8% |
| lm_head        | 0.316 | 2.2% |
| matmul_qkv     | 0.095 | 0.7% |
| rms_norm       | 0.049 | 0.3% |
| matmul_wo      | 0.035 | 0.2% |
| 기타           | 0.014 | 0.1% |
| **per-op 합**  | **14.13** | 100% |
| **wall (Decode ms/tok) 같은 세션** | **14.74** | sync gap 0.6 |

**모순 해소**:
- §12의 "13.23 > 12.45" 비교는 profile 세션 attention vs non-profile 세션 wall — 다른 측정 모드의 apples-to-oranges
- 같은 profile 세션 내: per-op 합 14.13 ≈ wall 14.74 (sync 0.6 차이) — **내부 일관성 OK**
- profile-events 자체가 +2.29 μs/n_kv 시스템 overhead 추가 (12.45 → 14.74)

**비-profile 세션 attention contribution 추정**: 13.23 × (12.45/14.74) = **11.18 μs/n_kv** = wall 12.45의 **90%**.

**llama.cpp profile 세션 Q1 alone slope**: **17.22 μs/n_kv** > 우리 13.23 (!). 그러나 비-profile wall은 5.70. 즉 llama.cpp는 profile 모드에서 더 크게 inflate (드라이버 차이) → **engine 간 profile-events 직접 비교 신뢰 불가**.

**산출물**: `.agent/research/microbench_flash_attn/option_c_op_slope_audit.md` (전체 분석 표)

**갭 분해 최종판** (비-profile 기준):
- Total 갭: 6.75 μs/n_kv
- Attention contribution: ~5.5 μs/n_kv (81%) ← 줄여야 할 것
- Non-attention contribution: ~1.25 μs/n_kv (19%) ← matmul_ffn/lm_head/matmul_qkv 등에서 분산

**다음 작업 권장 순서 갱신**: A (production 조건 microbench로 0.36 → 0.47 차이 isolate) → D (eviction-based 우회) → B (Snapdragon Profiler).

### 🆕 2026-04-14 21:50: Option A — production 조건 microbench 분해 → **결정적 발견**

**확장 매트릭스**: 4 variants × 2 layouts × 4 n_kv × 30 iters × 2 runs.

**HeadMajor per-token slope (μs/n_kv) 분해**:

| variant | mean slope | Δ from prev |
|---|---:|---:|
| Single × 28 (환산) | 10.48 | — |
| Repeat28 (back-to-back) | 9.98 | −0.50 (pipeline overlap) |
| +Mask (causal F16) | 10.30 | +0.32 (mask read) |
| +QVar (28-slot rotation) | 10.30 | 0.00 (no effect) |

**결과 1**: production attention slope 13.23 vs microbench-best 10.30 = **2.93 μs/n_kv 차이**가 production 환경의 17개 intervening ops (FFN matmul L2 thrashing 등)에서 옴. **수정 어려움 — 구조적**.

**결과 2 (더 중요)**: microbench-best 10.30 자체가 llama.cpp 추정 attention slope (~4.5) 대비 **2.3× 느림**. 즉 production 환경 노이즈를 0으로 제거해도 우리 커널이 본질적으로 더 느림. **Phase A의 "byte-identical 결론"이 실측과 모순 — 재검토 필요**.

**Phase A 누락 가능성**:
1. 실제 dispatch 되는 kernel 파일이 다름 (build variant)
2. global/local work size, subgroup 구성 차이
3. specialization constants 차이
4. vendor-specific 컴파일 옵션 차이
5. 드라이버 컴파일 결정 차이 (source identical even if asm differs)

**산출물**: `.agent/research/microbench_flash_attn/option_a_production_decomp.md`

### 🚀 다음 세션 엔트리 포인트 (재정비 v3)

A 결과로 "production 환경 overhead 2.93은 줄이기 어렵고, 진짜 갭은 커널 자체 (10.30 vs 4.5)"가 확정됨. Phase A 재검토 + B (Snapdragon Profiler)로 kernel-level isolation이 정공법.

**A1 (저비용, Phase A 재검토)**: researcher에게 다음을 위임
- Galaxy S25 디바이스에서 실제 llama.cpp 빌드가 사용하는 kernel binary 추출 (clGetProgramInfo + decompile)
- 우리와 같은 입력 (Qwen 2.5-1.5B, n_kv=2047, decode)에서 llama.cpp가 enqueue 하는 정확한 dispatch parameters (gws/lws/args) 캡처
- 두 kernel의 컴파일된 ISA 직접 비교

**A2 (고비용, B로 진행)**: Snapdragon Profiler trace
- 두 엔진의 attention dispatch만 격리하여 SP에서 비교
- L1/L2 hit rate, register usage, occupancy, stall reason 직접 측정
- 명확한 bottleneck class 식별 (memory-bound vs compute-bound vs occupancy)

**D (저비용, 갭 우회)**: 정확도 trade-off 수용 가능 시
- 이미 구현된 Sliding/H2O/D2O eviction을 long-context decode에 적극 적용
- effective n_kv 감소 → context 비례 갭 자동 감소
- 측정: `--eviction-policy sliding --eviction-window 1024` 에서 long context TBT 비교

### 🆕 2026-04-14 22:00: Option A1 — Phase A 재감사 (researcher 위임) → **결정적 발견 2개**

**산출물**: `.agent/research/2026-04-14_a1_phase_a_reaudit.md`

**가설 verdict (H1~H5)**:
| 가설 | 결론 | 근거 |
|------|------|------|
| H1 (잘못된 파일 비교) | REJECT | 같은 파일, 벤더 분기 없음 |
| H2 (dispatch 다름) | REJECT | gws/lws/40 args 모두 동일 |
| H3 (컴파일 옵션 다름) | REJECT | 의미적 동일 |
| H4 (드라이버 컴파일 차이) | UNDETERMINED | binary 추출 미수행. source 자체가 다르므로 moot |
| H5 (단일 vs 다중 kernel) | REJECT | 양쪽 모두 단일 dispatch |

**진짜 발견 1: Phase A는 prefill만 비교했고 Q1은 비교 안 함. 비교해보니 우리 Q1이 더 고도화됨**:

| 지점 | 우리 Q1 | llama.cpp Q1 |
|------|---------|--------------|
| kernel 속성 | `REQD_SUBGROUP_SIZE_64` (qcom) | 없음 |
| m_i / l_i / o_acc reduce | `sub_group_reduce_*` (barrier-free) | SLM tree + 6+6+(DV_VEC × 7) barrier |
| **총 barrier (DV=128)** | **0** | **236** |
| SLM 사용 | 0 | ~1.25 KB |

우리는 B-4 sprint에서 의도적 교체. **이론상 우리가 빨라야 함**.

**진짜 발견 2: "llama.cpp attention 4.5 μs/n_kv"는 phantom target — 직접 측정된 적 없음**:
- 4.5 = wall 5.70 (non-profile) × 0.8 추정값
- llama.cpp attention 직접 측정 (Phase B attn-event): **17.61 μs/n_kv** > 우리 13.23
- attn-event 빌드의 wall = 18.03 (4.79와 3.77× 차이) → Adreno 드라이버에서 `CL_QUEUE_PROFILING_ENABLE` 자체가 flash attn에 막대한 패널티
- 양쪽 다 profile overhead 포함 비교에서 **우리(13.23)가 llama(17.61)보다 4.4 μs/n_kv 더 빠름**
- "우리가 2.3× 느리다"는 evidence가 **현재 어떤 측정 조건에서도 재현된 적 없음**

### 🚀 다음 세션 엔트리 포인트 (재정비 v4)

A1 결과로 **갭의 실재 자체가 의심스러움** 상태. 더 진행 전 baseline 검증 선행.

**최우선 (cross-run 결정적 검증)**:
- llama.cpp Q1 kernel source를 우리 microbench harness에 import해서 같은 디바이스 같은 입력으로 직접 측정
- 우리 Q1 vs llama.cpp Q1 단일 kernel μs 비교 → **결정적 답**
- 우리가 빠르면: §12 갭 자체가 attention 외부 원인. attention 최적화 그만.
- 우리가 느리면: 우리 B-4 최적화가 Adreno에서 손해. 원복 검토.

**병행 (B-4 정당성)**:
- 우리 Q1 `sub_group_reduce_*` → SLM tree-reduce 임시 revert해서 production slope 변화 측정.

**보류 (Snapdragon Profiler)**:
- 갭 실재 미확인이라 대상 모호. cross-run 후 결정.

**계속 유효 (D)**:
- Eviction은 갭 실재 무관 독립 전략.

### 🚀 다음 세션 엔트리 포인트 (이전 v3, 보존용)

KV stride 가설까지 기각된 시점, 남은 후보를 ROI 순으로:

**A (저비용, 직접적)**: microbench 확장 — production 조건 모방
- 28× back-to-back dispatch (다른 layer 사이 KV 무효화 시뮬레이션)
- 실제 mask 버퍼 전달 (현재 NULL)
- Q값을 랜덤하게 매 iter 갱신 (production은 매 layer 다른 Q)
- 목적: microbench 0.36 vs production 0.47 의 **0.11 μs/n_kv per-layer 차이** (= 28× = 3.1 μs/n_kv per-token)가 어디서 오는지 isolate
- 이 차이를 재현하면 production overhead의 26% 설명 → 그 안에서 추가 optimization 여지

**B (고비용, 결정적)**: Snapdragon Profiler — 직접 shader trace
- L1/L2 hit rate, occupancy, register spill 직접 측정
- llm_rs2 vs llama.cpp attention kernel 정확히 어떤 단계에서 stall 하는지

**C (중간 비용, 새 방향)**: **attention 외부** 의심
- production attention slope 13.23 vs TOTAL 12.45 자체가 모순 (attention > total)
- 비-attention ops (RMSNorm growing-Q, KV scatter F32→F16, residual add) 중 일부가 context와 음의 상관을 가지거나, attention 측정에 다른 op이 섞여 들어왔을 가능성
- 검증: production --profile-events에서 op별 slope 합이 TOTAL과 일치하는지 재계산

**D (방향 전환)**: long-context 갭 자체를 우회
- 현재 갭은 long context (4K+)에서 ~33ms/tok = TBT 60% 증가
- KV cache eviction (이미 구현된 H2O/D2O/Sliding) 적극 활용으로 effective n_kv 감소 → 갭 자체 회피
- 정확도 trade-off 수용 가능 시 가장 빠른 win

### ✗ 최종 기각 가설 (누적, 6개)
1. Zero-copy KV
2. llm_rs2 내부 HeadMajor/SeqMajor (같은 head 연속)
3. Flash attn kernel 바이트 정적 분석
4. Launch count / per-launch host work context 비례
5. "우리 attention이 실제로 더 빠름" (Phase B 재해석)
6. **KV stride layout 차이 (HeadMajor 256B vs llama-view 512B)** ← 신규 기각

### 🎯 재확정 사실 (변화 없음)
- llm_rs2 decode TOTAL slope: 12.45 μs/n_kv
- llama.cpp decode TOTAL slope: 5.70 μs/n_kv
- 갭 slope: ~6.75 μs/n_kv
- llm_rs2 attention slope (--profile-events): 13.23 μs/n_kv
- microbench 단일 layer attention slope: **0.36~0.39 μs/n_kv** (신규)

---

## 12-prev. 이전 세션 가이드 (2026-04-14 16:15 — Context scaling 갭 실증 + **3개 가설 기각**)

### 🆕 2026-04-14 심야 세션: Baseline 재측정 — **오후 값 재현성 확정, attn-event 패치 artifact 확정, attention 귀책 복원**

**목적**: 오후(4.79 μs/n_kv) vs 밤 Phase B(18.03 μs/n_kv)의 llama.cpp TOTAL slope 모순 규명.

**재측정 결과 (단일 세션, 240s 쿨다운, 3 엔진 × 4 context)**:

| ctx | llm_rs2 | llama-orig | llama-attnevt |
|---:|---:|---:|---:|
| 256 | 30.80 | **29.97** | 35.76 |
| 1024 | 38.90 | **32.73** | — |
| 2048 | 100.13 ⚠ thermal | **36.04** | — |
| 6k | 110.65 ⚠ thermal | **53.99** | — |

**확정 결론**:
1. **오후 llama.cpp baseline 재현성 OK** — llama-orig slope **5.70 μs/n_kv** (오후 4.72와 ±1 μs 범위)
2. **attn-event 패치가 TOTAL +5.79 ms/tok 오버헤드** (ctx=256 orig 29.97 vs attnevt 35.76) — 28 layer × 31 token × ~6.67 μs/wait 계산과 일치. Phase B의 llama.cpp TOTAL 18.03 μs/n_kv는 **이 패치 artifact**
3. **따라서 Phase B의 llama.cpp attention slope 17.61 μs/n_kv도 inflated 빌드 측정** — 패치가 attention kernel 주변 GPU idle을 만들었을 가능성, 즉 event 시간도 실질적으로 늘어났을 수 있음
4. **올바른 비교**: llama.cpp attention slope는 TOTAL 5.70 μs/n_kv **이하**. llm_rs2 attention 13.23은 실제로 **≥ 2.3× 가파름** → Option 2B 원래 결론 **복원**

**Phase A 결론과의 핵심 모순 지속**:
- Phase A 정적 diff: 두 커널 K-loop **바이트 단위 동일**, 컴파일 옵션 동일, dispatch 동일. 우리 B-4 reduction이 오히려 유리해야 함.
- 실측: 우리가 2.3× 더 느림.
- **가설 (Phase A가 놓친 것)**: host가 주는 KV stride 차이가 Adreno 메모리 계층에서 실질적 성능 차이 유발:
  - llm_rs2 HeadMajor `k_nb1 = head_dim*2 = 256B` (한 head 내 pos 연속)
  - llama.cpp permute view `k_nb1 = head_dim * n_kv_heads * 2 = 512B` (pos 사이에 다른 kv_head 데이터 interleave)
  - 두 layout이 동일 K 데이터를 다른 access pattern으로 읽음 → L1/L2 prefetcher 효율 차이 가능
- llm_rs2 내부 SeqMajor A/B에서 기울기 0 차이(§12 원본)는 우리 두 layout 모두 "한 head 연속" 유지하는 구조라서 차이가 없었던 것. **llama 동등 stride는 "pos 사이에 head interleave"라는 제3의 layout**.

**측정 실패 영역**:
- **llm_rs2 ctx=2048/6k는 thermal 오염**: prefill도 오후 대비 10~52% 느려짐 (누적 열 + 쿨다운 부족). slope 분석에는 오후 값 그대로 사용 (12.45 μs/n_kv 유효).
- **llama-attnevt 1024/2048/6k 수집 실패**: 스크립트가 `llama-cli-attnevt`를 사용했으나 이 바이너리는 `-no-cnv` 미지원 → interactive mode 감옥. 정확한 바이너리는 `llama-completion-attnevt` (디바이스 `/data/local/tmp/`에 존재). 다음 세션 재측정 필요 시 이것으로.

### 🚀 다음 세션 엔트리 포인트 (최종)

**최우선 (Option 4, 신규, ROI 명확)**: **llama-permute-view 등가 stride 실험**
- llm_rs2 KV를 제3의 layout `[1, cap, kv_heads, head_dim]` (pos-major, kv_head 2차)으로 변경하여 **llama.cpp와 동등한 stride**로 attention kernel 호출
- 예상: 이 layout에서 attention slope이 llama.cpp 수준(~5 μs/n_kv)으로 수렴하면 **KV stride가 진짜 원인** 확정 → 프로덕션 KV store를 이 layout으로 전환
- 공수: 중간 (KV store/scatter/eviction 전 경로 수정). Phase A researcher가 작성한 stride 테이블을 시작점으로 사용.
- 파일럿 스킵 가능: KV 저장 자체를 안 바꾸고 attention kernel에 **2 stride 변형** 버전만 추가하여 A/B 측정이 더 저비용일 수 있음.

**차선 (조사 완결)**: attn-event 패치를 **kernel-outside wait만 수집**으로 개선하여 llama.cpp attention 진짜 GPU 시간 확정
- 공수: 패치 수정 + 동일 세션 bench (1~2시간)
- 목적: llm_rs2 13.23 vs llama.cpp attention의 **정확한 비율** 확정 (현재는 상한만 5.70)

### ✗ 최종 기각 가설 (누적)
1. Zero-copy KV (기각, production device-local)
2. llm_rs2 내부 HeadMajor/SeqMajor (기각, 같은 "head 내 pos 연속" 패턴이라 무차별)
3. Flash attn kernel 바이트 정적 분석 (기각, 동일)
4. Launch count / per-launch host work context 비례 (기각, 506 dispatches context-invariant)
5. "우리 attention이 실제로 더 빠름" (Phase B 결론; 기각, attn-event 패치가 llama 쪽 inflated)

### 🎯 재확정 사실 (누적)
- llm_rs2 decode TOTAL slope: **12.45 μs/n_kv** (오후 재측정)
- llama.cpp decode TOTAL slope: **5.70 μs/n_kv** (오후 + 심야 재현)
- 갭 slope: **~6.75 μs/n_kv** (기존 7.73에서 소폭 하향, 오차 범위)
- llm_rs2 attention slope: **13.23 μs/n_kv** (--profile-events CL event, 순수)
- attention이 llm_rs2 context-propotional 증가의 93% 지배 (Option 2B)
- Phase A 정적 동등 결론은 유효하나 **host stride 차이는 미조사 영역**

### 📂 오늘 세션 산출물
- `.agent/research/2026-04-14_option_3_phase_a_flash_attn_diff.md` (정밀 diff)
- `.agent/research/2026-04-14_option_3_phase_b_llamacpp_attn_event.{md,txt,patch}` (attn-event 패치 + 측정)
- `.agent/research/baseline_rerun_raw/` (12 run 원본 로그; attnevt 로그는 interactive 감옥으로 부풀려짐, 압축 권장)
- llama.cpp 디바이스 바이너리: `llama-cli-orig` (정상), `llama-completion-attnevt` (정상), `llama-cli-attnevt` (`-no-cnv` 미지원, 재측정 금지)

---

### 2026-04-14 밤 세션: Option 3 Phase A+B (원본, ★ Phase B 일부 결론 뒤집힘)

**Phase A (researcher)**: 두 커널 K-loop inner body **바이트 단위 동일**. 컴파일 옵션, dispatch 동일. KV layout은 오히려 llm_rs2 HeadMajor가 cache-friendlier. B-4 subgroup reduction으로 이론상 우리가 더 빨라야 함. **드롭인 포팅 기대효과 0**.

**Phase B (senior-implementer)**: llama.cpp에 CL event 계측 패치(`ggml-opencl.cpp` 107줄, 비커밋 working tree) → `flash_attn_f32_f16_q1` 단독 μs 직접 측정:

| n_kv | llm_rs2 μs/tok | llama.cpp μs/tok (신규 event) | 비교 |
|---:|---:|---:|---:|
| 258  | 5,691  | 9,964  | 우리 43% 빠름 |
| 1025 | 13,741 | 22,374 | 우리 39% 빠름 |
| 2047 | 18,456 | 41,527 | 우리 56% 빠름 |
| 4472 | 61,082 | 83,779 | 우리 27% 빠름 |

- **llm_rs2 attention slope: 13.23 μs/n_kv**
- **llama.cpp attention slope: 17.61 μs/n_kv**
- **갭 −4.39 μs/n_kv (우리가 오히려 가파르게 낮음)**

**방법론 대발견**: profile-build CSV 값(`cmd_end − cmd_start`)과 non-profile-build CL event 값이 **±1.7%로 일치**. Option 2B에서 전제한 "profile-build는 clFinish 오버헤드로 부풀려짐" 가정이 **무효**. 이로 인해 Option 2B의 역산("우리 attention이 gap의 168%") 결론 **전면 뒤집힘**. Option 3 (커널 포팅) **완전 기각**.

**핵심 모순**: 본 벤치의 llama.cpp eval_time 기반 TOTAL decode slope ≈ **18.03 μs/n_kv** (35.84→111.79 ms/tok, 4214 n_kv). **기존 baseline 4.79 μs/n_kv와 완전 모순**. 즉 기존 "갭 7.87 μs/n_kv" 수치의 전제 자체가 의심스러움. 우리 TOTAL 12.45 μs/n_kv는 재측정으로 일관되게 재확인되었으나, llama.cpp TOTAL는 **오후 벤치와 밤 벤치가 다른 값**을 보임. 쿨다운/thermal/빌드 옵션 차이 가능성.

### 🎯 다음 세션 우선 작업

**Step 1 (필수)**: **llama.cpp baseline decode TOTAL slope 재측정**
- 방법: `build-android-attn-event/bin/llama-completion-attnevt` 및 **원본 `build-android/bin/llama-cli`** 두 빌드 모두로 같은 4 context 벤치. 쿨다운 240s + thermal 0. `--no-display-prompt` 옵션 일관성 확인.
- 목표: llama.cpp TOTAL slope이 실제로 몇 μs/n_kv인지 확정. 만약 18 μs/n_kv면 **기존 갭 7.87 μs/n_kv 자체가 아티팩트**, 만약 5 μs/n_kv면 llm_rs2는 진짜 2.5× 더 느림 → attention 외부 원인 탐색으로 진행.

**Step 2 (Step 1 결과가 18 μs/n_kv면)**: baseline 아티팩트 확인. **갭 없음 결론 → long-context 최적화 작업 종결**, 다른 우선순위로 이동.

**Step 3 (Step 1 결과가 5 μs/n_kv면)**: attention은 모순(우리 13.23 > llama.cpp 17.61 측정값) → 측정 세션 간 비교 불가 시그널. 같은 세션에서 TOTAL + attention 동시 측정해야 의미. 단일 세션 내에서 둘 다 측정하는 통합 벤치 필요.

### ✗ 최종 기각 가설 (누적)

1. Zero-copy KV (기각, production device-local)
2. KV layout HeadMajor/SeqMajor (기각, slope 0.00 차이)
3. Flash attn K-loop inner 정적 구조 (기각, 바이트 동일)
4. Launch count / per-launch host work context 비례 (기각, 506 dispatches context-invariant)
5. **Attention slope 차이** (기각, 직접 측정 시 오히려 우리가 빠름)

### 🧹 MEMORY.md 정정

"long-context attention optimization" 관련 진행 상태는 **"진행 중, 갭 실체 자체 재검증 단계"**로 업데이트. "우리 attention이 느리다"는 기록은 제거.

### 산출물 (Phase B)
- `.agent/research/2026-04-14_option_3_phase_a_flash_attn_diff.md`
- `.agent/research/2026-04-14_option_3_phase_b_llamacpp_attn_event.{md,txt,patch}`
- `.agent/research/phase_b_raw/attnevt_ctx{256,1024,2048,6k}.log`
- llama.cpp 디바이스 바이너리 `/data/local/tmp/llama-completion-attnevt`

---

### 2026-04-14 저녁 세션: Option 2b 완료 (★ 역산 결론 무효, 절차 기록만 보존)

**attention GPU μs/n_kv 기울기 측정**:
| n_kv | llm_rs2 attn μs/tok |
|---:|---:|
| 258 | 5,691 |
| 1025 | 13,741 |
| 2047 | 18,456 |
| 4472 | 61,082 |

- **Attention slope: 13.23 μs/n_kv** (CL event profile-events 정확 측정)
- **Attention이 llm_rs2 per-op 전체 기울기의 93% 지배**. 비-attention ops (matmul_qkv/wo/ffn, rms_norm, kv_update, …) 전부 < 0.5 μs/n_kv = context-invariant
- **gap slope 7.87 μs/n_kv를 attention 단독으로 168% 설명** (overshoot)
- llama.cpp 절대값은 `GGML_OPENCL_PROFILING=ON`의 clFinish 오버헤드로 부풀려짐 (fattn 82ms > actual total 50ms), 직접 비교 불가하나 **우리 기울기 자체가 이미 갭 전량 설명**

**결론**: 갭의 거의 전부가 `flash_attn_f32_f16.cl` Q1 커널 내부 context-proportional 경로에서 발생. **Option 3 (drop-in llama.cpp kernel 포팅) 추진 근거 확보**.

**산출물**:
- `.agent/research/2026-04-14_option_2b_attention_slope.md` (분석)
- `.agent/research/2026-04-14_option_2b_attention_slope.txt` (벤치 로그)
- `.agent/research/option_2b_raw/` (8 원본 run + 4 llama CSV)

---

### 2026-04-14 저녁 세션: Option 1 (launch count) 정적 기각

**정적 분석만으로 기각 완료, 빌드/배포 불필요**.

Plan decode 경로 (`engine/src/backend/opencl/plan.rs:322 execute()`) 토큰당 dispatch 수:
- per-layer `steps_pre_kv` 9 (RMSNorm + QKV matmul+bias ×3 + RoPE Q/K) + `kv_update` 1 + `attention` 1 (**flash 단일 dispatch**) + `steps_post_attn` 7 (wo + add_rms_norm + gate + up + silu_mul + down + add) = **18**
- Qwen 2.5-1.5B 28 layers × 18 + final_norm 1 + lm_head 1 = **506 dispatches/token**

**핵심**: 이 수는 **전부 context-invariant**. Flash attention은 n_kv 길이와 무관하게 1 dispatch (K-loop은 커널 내부). KV scatter는 write_pos 1개만 dynamic. `DynamicArg`는 7개 스칼라(`start_pos/cache_seq_len/write_pos/kv_capacity/res_pos/q2_tokens/res_tokens`) 고정 수량.

즉 token당 launch count도, per-launch set_kernel_arg host work도 **context에 비례하지 않는다**. 7.7 μs/tok × context 기울기(Qwen 실측)는 launch overhead(7.8 μs × 506 ≈ 4 ms, 고정)로 **구조적으로 설명 불가**.

**결론**: §12 원본 "가능성 2" (launch count or per-launch host work의 context 비례 증가) **기각**. 남은 후보는 **가능성 1 / 3 / 4** (GPU 내부 stall, host loop, per-iter instruction level).

다음 세션 권장 경로:
- **Option 2b (신규, 저비용 권장)**: `--profile-events`로 **attention op GPU μs의 context 기울기**만 직접 측정. profile-events는 양쪽 엔진 모두 동일한 sync 오버헤드를 받으므로 기울기 비교는 유효. attention-only가 7.7 μs/tok 기울기의 몇 %를 설명하는지 정량 → flash attn 내부 기각/확정.
- **Option 3 (확정적)**: flash attn drop-in 실험 (우리 kernel ↔ llama.cpp kernel swap).
- **Option 2 (장비)**: Snapdragon Profiler.

---

### 원본 §12 내용 (2026-04-14 16:15)


### 🎯 이번 세션(오후) 최종 상태

**Context scaling 실측으로 갭의 구조가 완전히 확정됨**. 초기 제안 가설 2개 모두 기각. 파일럿 변경은 폐기 (워킹트리 revert 완료).

### 🔥 이번 세션 핵심 발견: Short에서 갭 ≈ 0, Long에서만 context-비례로 누적

Qwen 2.5-1.5B Q4_0 + F16 KV, Adreno 830, 쿨다운 240s + thermal 0 상태 실측:

| context | llm_rs2 ms/tok | llama.cpp ms/tok | 갭 |
|---:|---:|---:|---:|
| 258 | 30.81 | 30.19 | **+0.6** |
| 1025 | 38.80 | 32.78 | +6.0 |
| 2047 | 58.88 | 35.93 | +22.9 |
| 4472 | 83.30 | 50.08 | +33.2 |

**per-context-token 기울기**:
- llm_rs2: **12.43~12.45 μs/tok**
- llama.cpp: **4.72 μs/tok**
- 우리가 context 길이당 **2.64× 더 빨리 느려짐**

**의미**: Short(258)에서 갭이 사실상 0 → 커널 개별 속도, fusion, dispatch shape, launch overhead 등 **context-invariant 요인은 전부 동등**. 전체 갭 33.2 ms는 **context-proportional 한 축**(대부분 flash attention의 K-loop)에서 나옴. Attention 외 context 비례 경로는 decode에서 거의 없으므로 flash attn 쪽으로 범위 좁혀짐.

벤치 raw: `.agent/research/2026-04-14_decode_context_scaling_bench.txt`

### ✗ 기각된 가설 1 — "KV cache가 zero-copy라서 long에서 effective bandwidth 저하"

- 초기 추론: llm_rs2 `alloc_kv()`가 `UnifiedBuffer` (CL_MEM_ALLOC_HOST_PTR)로 host-backed 할당, llama.cpp는 전부 `CL_MEM_READ_WRITE` device-local → long context에서 GPU L2 miss 시 우리만 host RAM 왕복.
- **실제 코드 확인 (`generate.rs:564~582`, `memory.rs:59~75`)**: `--zero-copy` 플래그 없으면 `effective_zero_copy=false` → `alloc_kv()`가 `self.alloc()`으로 fallback → `CL_MEM_READ_WRITE` (device-local). **Qwen production 벤치는 이미 llama.cpp와 동일 flag로 할당 중**. 가설 전제가 성립 안 함.

### ✗ 기각된 가설 2 — "KV layout HeadMajor가 SeqMajor보다 long context에서 L2 재사용 불리"

- senior-implementer 파일럿: `--kv-layout seq` 전환용 코드 4파일 수정 (+85/-34 LOC, 미커밋). 동일 4-context 벤치 2회 반복:

| n_kv | HeadMajor | SeqMajor |
|---:|---:|---:|
| 258 | 30.76 | 30.81 |
| 1025 | 38.85 | 38.96 |
| 2047 | 63.06 | 63.55 |
| 4472 | 83.15 | 83.17 |

- **기울기 0.00 μs/tok 차이** — 레이아웃 swap이 갭에 전혀 영향 없음.
- 이유: Qwen GQA=6, n_kv_heads=2, head_dim=128. 한 WG가 단일 KV head의 연속 포지션만 읽음 → HeadMajor에서도 K-loop read는 256B 연속. 포지션 간 stride 차이(256B vs 512B)가 Adreno L1/L2 line에서 의미 없는 수준.
- 파일럿 변경은 **전부 revert 완료** (4파일 워킹트리 깨끗).

### 🐛 파일럿 부산물: master SeqMajor 경로의 silent bug 발견 (미수정)

- `engine/kernels/simple_ops.cl`의 `kernel_kv_scatter_f32_to_f16`이 HeadMajor offset을 하드코딩 (`h * capacity * head_dim + write_pos * head_dim + d`).
- Plan 경로에서 `--kv-layout seq` 선택 시 decode silently garbage ("The quick brown fox jumpses line has? brom" 유형). 
- Production이 HeadMajor 기본이라 user-visible 영향 없음. 하지만 "SeqMajor 지원됨" 주장은 사실과 다름.
- MEMORY.md의 **"KVLayout enum, all code paths support both via strides (migrated 2026-03-09)"** 기록은 **정정 필요** — qcf/d2o/eviction/shift_positions_for_head 등 HeadMajor 하드코딩 19개 이상 존재.

### 🔬 Researcher 정적 분석 결과 (파일 미작성, 대화에만)

우리 `flash_attn_f32_f16.cl` Q1 vs llama.cpp 동일 파일 Q1 K-loop inner body 정밀 비교:
- **동등으로 확정된 영역**: FMA 구조, vector width(half4 load + float4 acc), SLM async copy(둘 다 미사용), image1d_buffer_t(둘 다 flash_attn에 미사용), sub_group_broadcast(둘 다 Q1에 미사용 — gemv_noshuffle.cl 전용), K-loop 내 mask/softcap 분기, inner unroll, 2-pass softmax, 컴파일러 옵션, dispatch shape, Q row load 횟수, per-thread state.
- **유리한 영역 (우리 앞)**: subgroup reduction (B-4), wavefront 고정 (qcom attribute).
- 유일하게 발견된 구조적 차이가 KV layout이었으나 실험 A에서 기각됨.
- **즉 정적 분석으론 per-iter 2.64× 갭의 원인을 특정 못 함**. 동적 측정(프로파일러) 필요.

### 🤔 남은 가능성 재정리 (다음 세션 시작점)

Short 갭이 0인 사실로 좁혀진 **context-proportional 원인** 후보:

**가능성 1 (최유력, 동적 증거 필요)**: Flash attn Q1 커널의 **GPU stall/occupancy/L2 behavior 차이**
- 정적 코드는 동등이지만 Adreno HW 상에서 wait/stall이 다를 수 있음
- **방법**: Snapdragon Profiler로 flash_attn_f32_f16_q1의 n_kv=258/1025/2047/4472 stall breakdown (memory stall, register spill, synchronization) 실측
- TODO §11 §가능성 A와 같지만 "profile 절대값 왜곡"은 **기울기 비교**에선 영향 적음

**가능성 2**: **Launch count 또는 per-launch host work의 context 비례 증가**
- 정적 분석 안 됐음. Decode token당 실제 `clEnqueueNDRangeKernel` 수를 두 엔진에서 직접 세어 context 길이별 변화 확인
- 수단: strace, cl_apitrace, 또는 event profiling에서 event 수

**가능성 3**: **우리 execute_plan 내부의 어떤 host-side loop이 context 비례 비용**
- 예: attention arg rebind, KV stride 재계산, eviction check, score collection 분기
- 수단: plan.rs `execute()` + `Step::FlashAttention` 케이스 감사, per-token profile로 CPU 시간 측정

**가능성 4**: **우리 flash attn Q1 내부의 어떤 per-iter 명령이 Adreno에서 실제로 더 비쌈**
- 정적 분석으론 안 드러난 명령 수준 차이 (예: half4 load 주소 계산 overhead, pointer arithmetic 차이)
- 수단: 우리 커널을 llama.cpp 버전으로 drop-in 대체 실험 (정합성 확보 후 갭 측정)

### ⛔ 더는 조사 가치 없는 영역

- Zero-copy KV (기각, production은 이미 device-local)
- KV layout HeadMajor/SeqMajor (기각, 기울기 무영향)
- Flash attn K-loop inner body 정적 구조 (researcher 정밀 비교 완료, 동등)
- Subgroup reduction, wavefront 고정, FMA, vector width, SLM, image 사용 (동등 또는 우리 유리)

### 📂 참고 자료 위치

- 본 세션 벤치: `.agent/research/2026-04-14_decode_context_scaling_bench.txt` (raw, 4 context × 2 engine + HeadMajor/SeqMajor A/B)
- 이전 조사 (여전히 유효): `.agent/research/2026-04-14_decode_{attention,matmul,kernel_fusion,microbench_plan,sync_audit}_*.md`

### 🚀 다음 세션 착수 옵션

**Option 1 (권장)**: **가능성 2** — launch count 직접 계측. 저비용, 한 세션 내. 결과가 "갭 정확히 X ms = N × 7.8 μs"면 launch overhead 확정, 아니면 가능성 3/4로.

**Option 2**: **가능성 1** — Snapdragon Profiler 설치 + flash attn Q1 stall breakdown. 장비 셋업 비용 있지만 확정적 답.

**Option 3**: **가능성 4** — 우리 flash attn .cl을 llama.cpp 버전으로 drop-in 대체 후 벤치. 정합성 확보(arg naming, stride 규약 맞추기)가 수십 분 소요. 결과가 갭 해소면 "우리 커널 내부 명령 수준 문제", 해소 안 되면 내부 무관.

**Option 4**: **작업 전환** — 33.2 ms 갭 수용, 다른 타겟으로 이동.

### 🧹 MEMORY.md 정정 할 일 (다음 세션 첫 step 권장)

- "KV Cache Architecture" 섹션의 "all code paths support both via strides" 문구 제거 또는 완화
- 실제 상태: Flash attn 본문은 stride-agnostic, 하지만 Plan scatter 커널, qcf/d2o/eviction/shift_positions_for_head 등은 HeadMajor 하드코딩

### 재현용 명령 (§11과 동일, 재확인용)

**벤치**:
```bash
adb shell "cd /data/local/tmp && ./generate \
  -m /data/local/tmp/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf \
  --prompt-file /data/local/tmp/prompt_{256,1024,2048,6k}.txt \
  -n 32 -b opencl --kv-type f16 --max-seq-len 6144 --ignore-eos"
```

**벤치 스크립트**: `/tmp/bench_context_scaling.sh` (이 세션에서 작성, 재사용 가능)

---

## 11. 이전 세션 시작 가이드 (2026-04-14 14:30 — Decode 갭 분석 6연속 네거티브, 작업 중단)

### 🎯 세션 최종 상태

**Decode 갭 30.6 ms/tok (우리 82.3 ms vs llama.cpp 51.7 ms = 우리 63% 수준) 확정**. 원인 조사 6연속 네거티브, 추가 조사 ROI 낮음 판정으로 **일단 중단**. 다음 세션에서 이어 받을 수 있도록 조사 결과 전량 보존.

### 현재 master 상태
- HEAD: `1d8b2ef docs(skills/worktree): add cwd-in-target guard + stale entry recovery` (Gemma 3 4B 지원 작업이 06212ab 이후 머지됨)
- Phase 1 커밋 `06212ab feat(opencl): event-based per-op profiling (--profile-events)` — 본 세션 산출물, 메인 브랜치 merged
- 빌드: `cargo check` 클린, Android aarch64 release OK
- 디바이스(/data/local/tmp): 최신 `generate` 배포됨 (profile-events 기능 포함)

### 🔥 2026-04-14 오후 재확정 실측 (쿨다운 4분 + thermal 0 대기)

**안정 조건에서 두 엔진 완벽 일관**:
| 엔진 | Prefill tok/s | Decode ms/tok |
|---|---|---|
| llm_rs2 Run 1 | 102.2 | 82.19 |
| llm_rs2 Run 2 | 102.2 | 82.42 |
| llama.cpp Run 1 | 102.85 | 51.64 |
| llama.cpp Run 2 | 103.10 | 51.69 |

**갭 확정**:
- **Prefill**: 거의 parity (우리 99%)
- **Decode**: 30.6 ms 느림 (우리 63% 수준)

이전 §11의 "22 ms 갭"은 thermal-throttled llama.cpp 측정과 fresh llm_rs2 측정을 비교한 결과로 **잘못된 수치**였음. 정확한 갭은 **30.6 ms**.

### 갭 패턴 — Launch count 기반 가설의 증거

| | llm_rs2 | llama.cpp | 갭 |
|---|---|---|---|
| Prefill (큰 GEMM, 적은 launch) | 102 tok/s | 103 tok/s | 0.6% |
| Decode (작은 GEMV, 364 launches/token) | 82 ms | 52 ms | **59%** |

Launch 수에 비례한 갭 증가 → per-kernel launch overhead 또는 per-launch host work가 원인이라는 가설. **하지만 실측으로 반증됨** (아래 §Phase 2 참고).

---

### ✅ 이번 세션 성과

#### 1. Event-based per-op profiling 구현 (`06212ab`)
- CLI flag `--profile-events` 추가
- OpenCL `CL_QUEUE_PROFILING_ENABLE` + event capture로 synchronize 없이 per-op GPU μs 측정
- 6파일 +582/−309 LOC
- llama.cpp `GGML_OPENCL_PROFILING=ON` 빌드와 apples-to-apples 비교 가능

#### 2. llama.cpp profile 빌드 + CSV 측정 확보
- 재빌드: `/home/go/Workspace/llama.cpp/build-android-profile/bin/llama-cli`
- 측정 결과: `/tmp/llama_cl_profiling.csv` (79,274 이벤트)
- decode 구간 per-op 분해: attention 85ms, FFN 12ms, lm_head 10.6ms 등

#### 3. Microbench 도구 추가 (untracked)
- `engine/src/bin/microbench_launch.rs` — Rust ocl::core로 noop kernel 10000회 launch
- `experiments/benchmarks/microbench_launch.c` — 동일 로직 C raw OpenCL
- Adreno 830 실측: **Rust 7.93 μs vs C 7.82 μs (+0.11 μs 노이즈)**. ocl crate wrapper 오버헤드 없음 확인.

---

### 6연속 네거티브 조사 요약

| # | 가설 | 조사 결과 | 보고서 |
|---|---|---|---|
| 1 | attention 커널 느림 | ✗ 우리 Q1 flash attn가 llama.cpp보다 앞섬 (B-4 subgroup reduction) | `.agent/research/2026-04-14_decode_attention_llamacpp_adreno.md` |
| 2 | matmul_ffn/matmul_wo 미반영 기법 | ✗ Q4_0 GEMV 완전 동등, F16 GEMV 우리가 더 공격적 | `.agent/research/2026-04-14_decode_matmul_llamacpp_adreno.md` |
| 3 | kernel fusion 부족 | ✗ 대부분 갖춤. `add_rms_norm_oop` / `kv_scatter_f32_to_f16`은 우리가 앞섬 | `.agent/research/2026-04-14_decode_kernel_fusion_llamacpp_adreno.md` |
| 4 | per-layer `backend.flush()` × 32 | ✗ Qwen production은 `execute_plan` 경로 사용 → flush 애초에 호출 안 됨 | `.agent/research/2026-04-14_decode_sync_audit.md` |
| 5 | decode 경로 sync point audit | ✗ plan.rs 0건, backend method hot path 0건. llama.cpp와 동일 구조 | 위 같은 문서 + 수동 감사 |
| 6 | Rust ocl wrapper launch overhead | ✗ raw C와 0.11 μs 차이 (1.4%). 364 × 7.8 μs = 2.85 ms만 설명 | 위 microbench 바이너리 |

### 조사/측정 자료 전량 보존 위치

`.agent/research/`:
- `2026-04-14_decode_attention_llamacpp_adreno.md` — attention 기법 대조
- `2026-04-14_decode_matmul_llamacpp_adreno.md` — Q4_0/F16 GEMV 대조
- `2026-04-14_decode_kernel_fusion_llamacpp_adreno.md` — fusion 기법 대조
- `2026-04-14_decode_microbench_plan.md` — micro-bench 설계 (Phase 1/2/3/4)
- `2026-04-14_decode_sync_audit.md` — decode 경로 sync 감사

`/tmp/` (세션-local):
- `llama_cl_profiling.csv` — llama.cpp per-op 79k 이벤트
- `llama_decode_per_op.md` — llama.cpp 집계 표
- `llm_rs2_profile_events/*.json` — llm_rs2 event-based decode JSON
- `bench_gap_verify_v2_results.txt` — 5x5 초기 갭 측정 (thermal variance 문제 있음)
- `gpu_freq_v3_results.txt` — GPU 주파수 분포 (llm_rs2 1096 MHz vs llama.cpp 1051 MHz 평균)
- `freq_llm_rs2_r*.csv`, `freq_llamacpp_r*.csv` — raw freq traces
- `bench_gap_verify_v2.sh`, `gpu_freq_bench_v3.sh` — 벤치 스크립트 (재사용 가능)

---

### 📊 핵심 수치 정리 (다음 세션 참고용)

**Production 실측 (Qwen 2.5-1.5B Q4_0, Adreno 830, 4472 prefill tok)**:
- llm_rs2: Prefill 102 tok/s, Decode 82.3 ms/tok (12.2 tok/s)
- llama.cpp: Prefill 103 tok/s, Decode 51.7 ms/tok (19.3 tok/s)
- **Gap: Prefill 0.6%, Decode 30.6 ms (40%)**

**Profile 모드 (`--profile-events` / `GGML_OPENCL_PROFILING=ON`)**:
- llm_rs2: Decode 96 ms/tok, GPU kernel sum 92 ms (96.4%)
- llama.cpp: Decode 114.94 ms/tok, GPU kernel sum 111.56 ms (97.1%)
- 주의: CL_QUEUE_PROFILING_ENABLE이 per-kernel GPU HW time을 **부풀리는** 것으로 확인됨 (llama.cpp 115→52 ms compression 불가능). Profile 수치를 production 절대값으로 사용 금지 — **상대 %만 유효**.

**GPU 주파수 (production decode 중)**:
- llm_rs2 평균 1096 MHz (48% at 1200 MHz) — 더 공격적 사용
- llama.cpp 평균 1051 MHz (20% at 1200 MHz) — 여유 있게 사용
- 우리만 3% 시간 525 MHz로 드랍 (순간 throttling 또는 idle sleep)

**Raw OpenCL per-launch (noop kernel)**: 7.82 μs (Adreno 830 하한)

---

### 🤔 남은 미해결 가능성 (다음 세션 시작점)

6연속 네거티브 후에도 30.6 ms 갭은 실재. 남은 가능성:

**가능성 A — Production 실제 kernel GPU time 차이 (측정 미완)**
- Profile 모드가 왜곡되므로 production kernel sum은 알 수 없음
- **실험**: event-based profiling을 **최소 커널만** 선택해 켜기 (전체 activate가 아닌 sparse)
- 또는: production binary에서 `clEnqueueMarker` + `clWaitForEvents`로 특정 op만 측정
- 예상 가치: 높음 (정확한 갭 분포 확정)

**가능성 B — Launch count 차이**
- 우리 decode token당 launch 수를 정확히 세어 llama.cpp와 비교
- 우리 `execute_plan`의 step 수 × 각 step의 backend method 내부 kernel 수 = N
- llama.cpp의 `ggml_backend_opencl_graph_compute` iteration 수 + fusion 후 실제 enqueue 수 = M
- `N - M` × 7.82 μs = launch 수 차이로 설명 가능한 시간
- 예상 가치: 중간

**가능성 C — Per-launch host work 차이**
- 우리 backend method는 `get_cl_mem()` downcast chain × 4 variant 체크
- 각 launch마다 ~1 μs × 364 launches = 0.36 ms (미미)
- 하지만 `set_kernel_arg` 호출 수 차이는 측정 필요
- 예상 가치: 낮음

**가능성 D — UMA buffer cache coherence 문제**
- 우리는 zero-copy (CL_MEM_ALLOC_HOST_PTR) 사용
- 일부 버퍼가 CPU 포인터 통해 접근 → kernel 간 cache flush 강제 가능성
- llama.cpp는 device-local buffer 위주
- 예상 가치: 높음 (가설 검증 안 된 영역)

**가능성 E — OpenCL driver optimization hint**
- llama.cpp는 특정 kernel에 `cl_qcom_priority_hint` 같은 Qualcomm extension 사용 여부
- 우리가 놓친 driver-level 최적화 플래그
- 예상 가치: 중간

### 다음 세션 착수 옵션

**Option 1 (권장)**: **가능성 A** — production 경로에서 특정 op만 event profiling. `plan.execute` 내부에 선택적 event capture 추가.

**Option 2**: **가능성 D** — UMA cache coherence 감사. `CL_MEM_USE_HOST_PTR` / `CL_MEM_ALLOC_HOST_PTR` 사용 지점 전수 + llama.cpp의 버퍼 flag 정책 대조.

**Option 3**: **가능성 B** — launch count 직접 비교. llama.cpp `ggml-opencl.cpp` decode path + llm_rs2 `plan.execute` 한 iteration의 `clEnqueueNDRangeKernel` 호출 수 카운트.

**Option 4**: **작업 전환** — 30.6 ms 갭 수용, 다른 타겟으로 이동 (모델 범용성 확장, quality, 메모리 최적화 등).

---

### 🔥 기존 핵심 교훈 (여전히 유효)

**Adreno 830 DK=128 flash attention per-thread state 상한 = 32 float4** (A-3 B-1/B-4 성공, B-2/B-3 revert 이력).

**방법론**:
1. per-thread 배열 키우는 변경은 실측 전까지 부정적 가설
2. 대역폭 vs register 트레이드오프는 Adreno에서 거의 항상 register 쪽이 진다
3. state 불변 + reduction/barrier 비용만 줄이는 최적화가 ROI 가장 높음
4. llama.cpp가 구현하지 않은 기법은 Adreno 구조적 한계 가능성 우선 의심
5. 쿨다운 **240초 이상 + Thermal Status 0 확인** 후 측정 (120초는 부족)
6. **실측 게이트 = tok/s + 출력 품질 둘 다**
7. **Profile 모드 수치는 상대 비교만 유효**, 절대값은 production과 다름

### 다음 세션 진입 명령

```
.agent/todos/long_context_attention_optimization.md §11 읽고 Option 1부터 (또는 Option 4 결정)
```

### 재현용 명령

**벤치** (쿨다운 240초 + thermal 0 대기 후 안정 측정):
```bash
adb shell "cd /data/local/tmp && ./generate \
  -m /data/local/tmp/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf \
  --prompt-file /data/local/tmp/prompt_6k.txt \
  -n 32 -b opencl --kv-type f16 --max-seq-len 6144 --ignore-eos"
```

**llama.cpp 벤치**:
```bash
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp \
  ./llama-cli-orig \
  -m /data/local/tmp/Qwen2.5-1.5B-Instruct-q4_0.gguf \
  -f /data/local/tmp/prompt_6k.txt \
  -n 32 -ngl 99 -c 6144 --no-display-prompt \
  --temp 0.8 --top-p 0.9 --top-k 40 -no-cnv"
```

**Event-based profiling**:
```bash
adb shell "cd /data/local/tmp && ./generate \
  -m /data/local/tmp/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf \
  --prompt-file /data/local/tmp/prompt_6k.txt \
  -n 128 -b opencl --kv-type f16 --max-seq-len 6144 --ignore-eos \
  --profile-events --profile-dir /data/local/tmp/results/profile_events"
```

**Microbench (launch overhead)**:
```bash
adb shell "cd /data/local/tmp && ./microbench_launch"
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./microbench_launch_c"
```

### 실측 장비 상태 기록 (재현성)

- Device: Galaxy S25 (Adreno 830, Snapdragon 8 Elite for Galaxy)
- 쿨다운: **벤치 간 240초 + Thermal Status 0 확인 필수**. 120초로는 분산 큼.
- Qwen Q4_0 gguf on device: `/data/local/tmp/Qwen2.5-1.5B-Instruct-q4_0.gguf` (1011 MiB)
- llm_rs2 모델: `/data/local/tmp/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf`
- 프롬프트: `/data/local/tmp/prompt_6k.txt` (4472 tokens)
- llama.cpp 바이너리: `/data/local/tmp/llama-cli-orig` (production, profile OFF)
- llama.cpp profile 빌드: `/home/go/Workspace/llama.cpp/build-android-profile/bin/llama-cli` (필요시 디바이스 `llama-cli-prof`로 재배포)
