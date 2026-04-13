# Backlog — 미배정 작업

> 역할이 배정되지 않은 작업 대기열. PM이 우선순위 판단 후 역할별 TODO로 이동.

---

## [P0] Long context CPU attention 최적화 — 4K에서 llama.cpp 대비 35% 수준
- **Status**: TODO (설계+측정 완료, 구현 대기)
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: 4K context에서 llm_rs decode가 10.6 tok/s로 llama.cpp 30.5 tok/s의 35% 수준. Short context(~20)는 75%. 원인: standard 3-pass attention + head-parallel이 GQA 6:1에서 KV 중복 읽기(6배) + L2 thrash 유발. DRAM-bound가 되어 context 확장 시 급격 열화.
- **Acceptance Criteria**:
  - 4K context decode: 25+ tok/s (llama.cpp 대비 80% 이상)
  - Short context 회귀 없음 (22 tok/s 이상 유지)
  - 정확도 유지 (F16 NMSE < 1e-4, top-k match > 99%)
- **상세 계획**: `.agent/todos/long_context_attention_optimization.md`
- **구현 단계**: (1) Online Softmax (Step 1, 낮은 난이도) → (2) Flash Decoding KV split (Step 2, 중간 난이도, 메인 효과) → (3) CPU Flash Attention for prefill (Step 3, prefill O(n²) 해결)
- **주 수정 파일**: `engine/src/backend/cpu/neon.rs:235 attention_gen_f16_neon`
- **담당 권장**: senior-implementer (NEON + numerical algorithm)
- **측정 환경**: Galaxy S25 / Snapdragon 8 Elite / Qwen2.5 1.5B Q4_0 (`qwen2.5-1.5b-q4_0-v2.gguf`) / 6 threads
- **측정일**: 2026-04-13

---

## [P3] 다중 모델 사이즈 검증 테스트 매트릭스
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 다중 디바이스 포팅 완료
- **Description**: Llama 3.2 1B/3B 및 향후 7B/8B 모델에 대한 디바이스별 테스트 매트릭스 정의
- **Acceptance Criteria**: 매트릭스 문서, 디바이스별 최대 지원 모델 크기 명시
- **Notes**: 실제 테스트는 디바이스 확보 후 진행

## [P2] NVIDIA GPU OpenCL 추론 정확성 문제
- **Status**: DONE
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: |
  NVIDIA RTX 3090 Ti에서 OpenCL 백엔드로 추론 시 garbage 출력 발생.
  개별 커널(rms_norm, F16 matmul, softmax, half 읽기)은 pyopencl 단위 테스트에서 정확하나,
  전체 추론 파이프라인에서 garbage 발생. Q4 weight + F32 KV cache에서도 동일 → F16 커널 무관.

  ### 조사된 사항 (2026-03-24)
  - fallback 커널 컴파일: F32, Q4_0, Simple Ops, F16 모두 nosub 컴파일 성공
  - PoCL CPU OpenCL: 정상 추론 (subgroup 지원 → 원본 커널 사용)
  - 개별 커널 정확성: rms_norm, matmul_f16 모두 pyopencl 테스트 통과
  - `CL_MEM_ALLOC_HOST_PTR` (UnifiedBuffer): NVIDIA discrete GPU에서의 동작 미검증
  - `unified_buffer::test_map_write_unmap_cycle`: 호스트에서 panic 발생 (기존 이슈)

  ### 의심 원인 (우선순위순)
  1. UnifiedBuffer + CL_MEM_ALLOC_HOST_PTR의 NVIDIA 호환성 (버퍼 동기화/매핑)
  2. 커널 간 데이터 전달 시 GPU↔Host 메모리 일관성 문제
  3. nosub 커널 내 미세 인자 불일치 (dispatch parameter vs kernel expectation)
- **Acceptance Criteria**: NVIDIA GPU에서 CPU 백엔드와 동일한 coherent 텍스트 생성
- **Notes**: |
  - 환경: NVIDIA RTX 3090 Ti, OpenCL 3.0 CUDA, cl_khr_subgroups 미지원
  - F16 nosub fallback 커널은 구현 완료 (17b2763)
  - 디버깅 접근: UnifiedBuffer를 비활성화(use_zero_copy=false)하여 discrete GPU용 버퍼 할당으로 전환 테스트 권장

## [P2] Gemma 3 1B NVIDIA GPU 추론 실패
- **Status**: DONE
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: |
  Gemma 3 1B이 NVIDIA RTX 3090 Ti에서 `<unused6241>` 토큰만 생성.
  CPU에서는 정상 동작. Llama/Qwen은 NVIDIA에서도 정상.

  Gemma 3 특이사항:
  - head_dim=256 (Llama=64, Qwen=128)
  - `kernel_attn_gen_half`에서 `float out_local[256]` → 256 registers/thread
  - NVIDIA register limit (255) 초과 → spill to local memory → 가능한 정확성 문제
  - sliding_window=512 (로컬 어텐션)
  - gelu_pytorch_tanh 활성화

  회귀 테스트 baseline에서 확인됨 (735ba71).
- **Acceptance Criteria**: Gemma 3 1B NVIDIA GPU에서 coherent 텍스트 생성
- **Notes**: regression_test.py 3/3 FAIL (nvidia), 2/3 PASS (cpu)

## [P1] Manager ↔ Engine 프로토콜 이슈 (E2E 테스트에서 발견)
- **Status**: DONE
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: |
  2026-03-24 E2E 테스트(Manager + mock_engine, Unix socket)에서 발견된 이슈 목록.

  ### 1. Relief Model cold-start 문제
  relief model이 없으면 모든 예측이 `ReliefVector::zero()` → ActionSelector가 액션을 선택 불가.
  **수정 방향**: ActionSelector에서 observation_count==0인 액션에 대해 domain 기반 default relief를 반환하는 fallback 추가.

  ### 2. `~` 경로 미확장
  `ReliefModelConfig::default()`의 `storage_dir: "~/.llm_rs/models"`가 셸 확장 안 됨.
  **수정 방향**: `dirs::home_dir()` 또는 `std::env::var("HOME")` 기반 절대 경로로 확장.

  ### 3. main config `[policy]` 섹션 미사용
  `Config`에 `policy: Option<PolicyConfig>` 필드가 있으나, `load_policy_config()`는 `--policy-config` CLI 플래그만 읽음. main config의 `[policy.*]` 섹션이 무시됨.
  **수정 방향**: `--policy-config` 미지정 시 `config.policy`를 fallback으로 사용.

  ### 4. 단방향 소켓 (Manager가 Engine 메시지를 읽지 않음)
  `UnixSocketEmitter`는 write-only. Engine이 보내는 Capability/Heartbeat/Response를 Manager가 수신하지 않음.
  프로토콜 스펙(docs/37)은 양방향이나 구현은 단방향.
  **수정 방향**: UnixSocketEmitter를 양방향 `UnixSocketTransport`로 리팩토링, reader 스레드 추가, Heartbeat → Pipeline의 `engine_state` 갱신.

- **Acceptance Criteria**: |
  1. Seed model 없이도 cold-start에서 directive 생성 가능
  2. `~` 경로가 올바르게 확장됨
  3. main config의 `[policy]` 섹션이 인식됨
  4. Manager가 Engine Heartbeat를 수신하여 pipeline engine_state 갱신
- **Notes**: |
  - 이슈 #1, #2, #3은 독립적으로 수정 가능 (각각 소규모)
  - 이슈 #4는 아키텍처 변경 (UnixSocketEmitter → 양방향 transport). 설계 검토 후 진행
  - E2E 테스트 커맨드: `manager --transport unix:<sock> --policy-config <toml>` + `mock_engine --socket <sock>`
  - emit_initial 프로토콜 불일치는 수정 완료 (7895824)

---

# Spec-Implementation Divergence (2026-03-31 조사)

> spec/에 정의되어 있지만 코드에 구현되지 않은 항목. 우선순위 순.

## [P1] QcfEstimate 메시지 + RequestQcf 커맨드 구현
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: |
  MSG-014: EngineMessage에 QcfEstimate variant 없음.
  MSG-036b: EngineCommand에 RequestQcf variant 없음.
  Manager가 Critical 모드에서 QCF 비용 기반 액션 선택을 못 함.

  필요 구현:
  1. shared/src/lib.rs — QcfEstimate 구조체 + EngineMessage variant 추가
  2. shared/src/lib.rs — EngineCommand::RequestQcf variant 추가
  3. Engine — QCF 계산 후 QcfEstimate 전송 로직
  4. Manager — Critical 진입 시 RequestQcf Directive 발행 + 1초 타임아웃 수신
- **Acceptance Criteria**: Manager가 Critical 모드에서 RequestQcf → QcfEstimate 수신 → Lossy 액션 cost 반영
- **Notes**: 프로토콜 레벨 변경이므로 Architect spec 검토 필요. SEQ-090~098 참조.

## [P2] Manager 페이로드 크기 가드 추가
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: |
  PROTO-012: Engine측은 64KB MAX_PAYLOAD 검증 구현 완료.
  Manager측(unix_socket.rs, tcp.rs)의 read_engine_message()에 페이로드 크기 검증 없음.
  악의적/버그 Engine이 거대 페이로드를 보내면 OOM 위험.
- **Acceptance Criteria**: Manager가 64KB 초과 메시지를 거부하고 연결 유지
- **Notes**: 소규모 변경. manager/src/channel/unix_socket.rs:311, tcp.rs:299.

## [P2] Heartbeat/Response 타임아웃 구현
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: |
  SEQ-087: Manager가 Engine Heartbeat 부재를 감지하지 못함 (권장 3초).
  SEQ-088: Directive 후 Response 무한 대기 가능 (권장 500ms).
  Engine 장애 시 Manager가 대응 불가.
- **Acceptance Criteria**: Heartbeat 3초 미수신 시 Disconnected 전이. Response 500ms 초과 시 타임아웃 처리.
- **Notes**: 타이밍 상수는 Config로 설정 가능하게.

## [P2] KvStreaming 커맨드 정상 구현
- **Status**: DONE (2026-03-31)
- **Sprint**: backlog
- **Notes**: cc0b9ce — EngineCommand::KvStreaming → StreamingLLMPolicy 연결 완료

## [P2] KvMergeD2o 액션 추가
- **Status**: DONE (2026-03-31)
- **Sprint**: backlog
- **Notes**: ffce391 — Pipeline 재활용 설계, D2OHandler 수정 0줄

## [P3] MergeHandler 정상 구현
- **Status**: CANCELLED (2026-03-31)
- **Notes**: D2OHandler가 cosine merge를 이미 수행. 기능 중복으로 stub 삭제 (7742543)

## [P3] SparseHandler 정상 구현
- **Status**: CANCELLED (2026-03-31)
- **Notes**: 1B+2048ctx 타겟에서 실익 없음. stub 삭제 (7742543)

## [P3] EnergyConstraint 스펙-코드 Divergence 해소
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 없음
- **Description**: |
  MGR-ALG-015: 스펙은 raw battery_pct → 연속 pressure (m = clamp(1-pct/100, 0, 1) * 0.5).
  코드는 Level enum → 4단계 이산값 (Normal=0.0, Warning=0.55, Critical=0.80, Emergency=1.0).
  기능적으로 동작하지만 스펙과 다름.
- **Acceptance Criteria**: 스펙 수식대로 연속 변환하거나, 스펙을 현재 구현에 맞게 갱신
- **Notes**: 스펙 갱신이 더 현실적일 수 있음. Architect 판단 필요.

---

## [P3] ThermalCollector zone 패턴 매칭 auto-discovery
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 없음
- **Description**: 현재 `zone_types`는 exact match. substring/keyword 매칭으로 확장하면 다양한 장치 커버 가능
- **Acceptance Criteria**: contains 기반 패턴 매칭, 기존 exact match와 공존, 테스트 추가
- **Notes**: 필요성 미확정. 실제 다중 장치 배포 시점에 재평가

## [P1] Qwen CPU decode gap 해소 — matmul 외 원인 조사 필요
- **Status**: TODO (재정의됨)
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: |
  Qwen 2.5-1.5B CPU decode가 llama.cpp CPU 대비 +14-15% 느린 gap이 남아 있음.

  ### 이미 시도한 것 (모두 실측 효과 없음)
  1. **Native F16 FMA 전환** (`a9cd3cc`, 2026-04-11): FMLAL → FMLA .8H inline asm.
     Short −0.5% / long +0.3% (mean) — noise 범위. Commit 유지됨 (cleanup 가치).
     분석: `results/data/flash_attn_decode/thermal/FMA_ANALYSIS.md`
  2. **vfmaq_f16 intrinsic 포팅** (branch `feat/f16-intrinsic-gemv`, 2026-04-11,
     revert): nightly toolchain + `stdarch_neon_f16`. Disassembly 상 main loop
     36→30 instructions, load-to-use 거리 2→3-5로 명확히 개선. 그러나 실측 short
     동등, long 데이터 부족, **prefill +15-20% regression**. Net negative → revert.
     분석: `results/data/flash_attn_decode/thermal/INTRINSIC_EXPERIMENT.md`

  ### 학습된 것
  - **Kernel-level instruction scheduling 최적화는 실측에 반영되지 않음**: S25에서
    FMA GEMV는 이미 memory subsystem ceiling에 가까움. Disassembly 개선이 runtime
    개선을 보장하지 않는다.
  - **Nightly toolchain 전환은 숨은 cost 있음**: 포팅 대상이 아닌 prefill 경로에서도
    regression 관찰됨. LLVM/codegen 차이가 전체 binary에 영향.
  - 과거 S24 `b25bc19` 교훈 재확인: "inner loop optimizations (multi-row, prefetch,
    stride) have no effect because the bottleneck is DRAM bandwidth".

  ### 이제 필요한 것: 진짜 병목 찾기
  Kernel 최적화 루트는 exhausted. 다음 접근:

  1. **Per-op 프로덕션 프로파일링** (--profile 없이). `simpleperf`, `perf`, 또는
     수동 timestamp로 token당 어디에 시간이 쓰이는지 정확히 측정. matmul_ffn 외
     candidate: RMSNorm, attention softmax, sampling, thread dispatch, SpinPool
     overhead. **이 정보 없이는 추가 최적화가 다 hunch-driven이다.**
  2. **Thread pool dispatch overhead 측정**: SpinPool 자체의 per-chunk cost.
     llama.cpp threadpool과 어느 정도 차이 나는지.
  3. **Chunk size A/B** (llama.cpp 64 rows/chunk vs 우리 140 rows/chunk). 1줄 변경,
     빠른 실험 가치 있음.
  4. **Big.LITTLE affinity**: Long decode가 bi-modal (±5.8% spread)인 이유가
     Oryon Phoenix L/M 스케줄링 jitter일 가능성. Gap 축소보다 variance 축소.
  5. **Single-asm super-block** (stable-friendly 최후 옵션): 4 rows 모두 한 asm
     블록 안에서 explicit interleaving. 학습 결과(1) 기반으로 ROI 낮아 보이지만
     theoretical latency-hiding 경로를 완전히 소진할 마지막 카드.

  **(1)이 blocker** — 병목이 어디인지 모른 채로 (2)-(5) 시도는 또 다른 neutral
  실험이 될 위험.
- **Acceptance Criteria**: |
  - 먼저 (1) 프로파일링으로 per-op breakdown 확보 → 보고
  - 그 다음 가장 큰 op을 타겟으로 실측 최적화
  - 최종 목표: CPU decode short ≤ llama.cpp + 5%, long ≤ llama.cpp + 5%
  - V10 strict thermal isolation 프로토콜로 검증
- **Notes**: |
  - **시작점은 kernel이 아니라 measurement**. hunch에 기반한 kernel 변경은 금지.
  - Quality/Correctness는 `--greedy` byte-identical test로 보호
  - branch `feat/f16-intrinsic-gemv` 유지 (미래 참고용, merge 안 됨)
  - Device backup `/data/local/tmp/generate.fma-asm.backup` 유지
