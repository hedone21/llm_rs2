# Decode 경로 Host-Device Sync Audit

**작성일**: 2026-04-14
**계기**: Phase 1/2 event profiling 완료. GPU 커널 합계는 우리가 19 ms/tok 빠른데 production은 22 ms 느림 → 원인은 dispatch/pipelining.

## 핵심 발견

**llm_rs2가 llama.cpp에 비해 구조적으로 sync/flush가 많다**:
- **llama.cpp**: graph 전체를 in-order queue에 쌓고 마지막 logit 읽을 때 **단 1회 sync**
- **llm_rs2**: `forward_into` 경로는 per-layer `backend.flush()` × 2 = **token당 32회 clFlush** + `synchronize()` + blocking `read_buffer(logits)` = **최소 2회 clFinish**

**구조적 차이**: llama.cpp는 driver가 모든 kernel을 한 번에 보고 prefetch/launch pipelining 최적화. llm_rs2는 layer 단위 flush로 queue depth가 얕아지고 kernel 사이 idle gap 생김.

## 경로별 sync 수

| 경로 | clFlush/token | clFinish/token | blocking read/token |
|---|---|---|---|
| `execute_plan` (플랜 경로) | 0 | 1 | 1 |
| `forward_into` (fallback) | **32** | 1 | 1 |
| llama.cpp graph_compute | 0 | 1 | 1 |

**핵심 질문**: Qwen 2.5-1.5B 프로덕션 벤치가 어느 경로 사용 중인지 확인 필요.
- `generate.rs:3141` main loop는 plan 우선 시도 → Qwen에서 plan 빌드 성공 여부가 관건
- experiment 경로 (`:2145`)는 항상 `forward_into` → 32회 flush 발생

## Top 3 제거 후보

### 후보 1 (최우선): per-layer `backend.flush()` 제거
- **위치**: `engine/src/layers/transformer_layer/forward_gen.rs:195` (QKV 후), `:1100` (FFN gate/up 후)
- **근거**: llama.cpp 대응 없음. in-order queue에서 flush는 correctness에 불필요.
- **예상 ROI**: **10~20 ms/tok** (82.6 → 60~72 ms/tok, 갭 22 ms의 상당 부분)
- **구현**: 환경변수 가드 (`LLM_RS_DECODE_LAYER_FLUSH`) 기본 off로 전환
- **위험**: 낮음 (final clFinish + blocking read 남음, 정확성 안전)

### 후보 2: `generate.rs:3210` `synchronize()` 제거 (중복)
- **근거**: 바로 뒤 `read_buffer(blocking=true)`가 이미 queue drain
- **ROI**: 0.05~0.3 ms/tok
- **위험**: `--profile-events` 경로와 상호작용 고려 필요

### 후보 3: experiment 경로를 `execute_plan`으로 전환
- **근거**: experiment loop는 `forward_into`만 호출 → 32 flush 발생 → 불공정 벤치
- **ROI**: 10~20 ms/tok (experiment 측정)
- **위험**: importance collection / score accumulator 경로가 plan 미지원 가능

## Plan B 실험 프로토콜 (senior-implementer용)

### 사전 체크
**Qwen 2.5-1.5B 프로덕션 벤치가 plan vs forward_into 어느 경로 사용 중인지 확인** — stderr 로그, 플래그, 또는 해당 코드 경로에 printf.

### A/B 실험 (후보 1)

1. `forward_gen.rs:194-196, 1099-1101`에 환경변수 가드 추가
2. Galaxy S25 빌드/배포
3. 세 구성 측정 (동일 프롬프트, 20 토큰 × 5회):
   - A: 현재 (flush 유지)
   - B: `LLM_RS_DECODE_LAYER_FLUSH=0` (flush 제거)
   - C: `--profile-events` A/B — kernel sum 동일 확인
4. 성공 기준: B 중앙값 A 대비 ≥5 ms/tok 단축, 출력 품질 동일

### 추가 측정 권장
`--profile-events`로 kernel start/end 타임스탬프 → kernel 간 idle gap 계산 (SUBMIT-START 차이 = driver queuing latency). flush 제거 전/후 gap 감소 확인.

## 핵심 파일:라인

### llm_rs2
- Sync 구현: `engine/src/backend/opencl/mod.rs:2379` (read_buffer), `:2499` (write_buffer), `:2534` (synchronize), `:2539` (flush)
- Queue 생성 (in-order): `mod.rs:406-412`
- Decode layer sync: `forward_gen.rs:195, 1100` (**flush, 최우선 타겟**)
- Decode loop: `generate.rs:2161, 3210` (synchronize), `:3235` (read_buffer)
- Plan 경로 (sync-free): `engine/src/backend/opencl/plan.rs:322` (execute)

### llama.cpp
- graph compute: `ggml/src/ggml-opencl/ggml-opencl.cpp:3545-3588` (loop 내 flush 없음)
- get_tensor sync: `ggml-opencl.cpp:4936-5309` (blocking read + clFinish 1회)

## 후속 확장 조사

1. `CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE` Adreno 지원 여부 + dependency graph 기반 auto-pipelining
2. `execute_plan` 경로를 experiment/QCF/score accumulator로 확장
3. Token id write 비동기화 (ROI 매우 작음, 후순위)
