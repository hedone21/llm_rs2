# Tester TODO

> **역할**: 테스트 전략 수립, 테스트 케이스 설계, QA 검증, 품질 게이트 관리
> **소유 영역**: `engine/tests/`, `docs/14_component_status.md`, `scripts/update_test_status.py`, `.agent/rules/TESTING_STRATEGY.md`

---

## [P2] 범용 디바이스 배포/테스트 스크립트
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Device Registry 시스템
- **Description**: adb(Android)/ssh(Linux) 추상화된 배포 스크립트
- **Acceptance Criteria**: 단일 명령으로 다중 디바이스 테스트 실행, 결과 JSON 수집
- **Notes**: 기존 run_android.sh를 범용화

## [P2] Manager 서비스 통합 테스트
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Manager 서비스 구현
- **Description**: Manager + LLM E2E 연동 테스트
- **Acceptance Criteria**: 각 시나리오 통과, 플랫폼별 검증
- **Notes**: MockTransport 기반 호스트 테스트 + 실 디바이스 테스트 분리

---

## [P1] Gemma 3 1B 온디바이스 검증
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Rust Developer의 GEMMA-1.11 완료 후
- **Description**: Gemma 3 1B 모델의 온디바이스 추론 검증.
  - CPU backend: 생성 텍스트 품질, tok/s 측정
  - OpenCL backend: 생성 텍스트 품질, tok/s 측정
  - Llama 3.2 1B 회귀 테스트: 기존 모델 성능/정확성 변경 없음 확인
  - Qwen 2.5 1.5B 회귀 테스트: 기존 모델 성능/정확성 변경 없음 확인
- **Acceptance Criteria**:
  - CPU/OpenCL 양 백엔드에서 coherent 텍스트 생성
  - Llama/Qwen2 tok/s가 ±5% 이내 (회귀 없음)
  - test_backend 통과

---

---

# Resilience 실효성 검증 (impl-request-v13)

> **목표**: madvise 정밀화 + chunked prefill + offload TCP 수정 후, Mode A/B/C 생존율 분리 재현
> **요청 문서**: `papers/pact2026/plan/impl-request-v13.md`
> **타겟 모델**: Qwen 2.5 1.5B (주력), Gemma 2 2B (보조)
> **디바이스**: S25 (12GB RAM)
> **선행 조건**: Rust Developer의 RESIL-1~7 완료

---

## [P0] RESIL-T1. madvise 정확성 검증
- **Status**: DONE
- **Sprint**: current
- **호스트**: ✅ 유닛 테스트 3개 통과
- **온디바이스** (S25, Qwen 2.5 1.5B): ✅ 완료 (2026-03-27)
  - H2O eviction 트리거: 115→86 tokens (29 evicted)
  - 로그: `released 16384 bytes (pos=86, hwm=115, cap=128)` — hwm 기준 범위 사용 ✅
  - 계산: 29 slots × 128dim × 2bytes(F16) × 2heads = 14,848 → page-aligned 16,384 ✅ 일치
  - 6 eviction cycles × 28 layers = 168 release 호출 정상

## [P0] RESIL-T2. Chunked Prefill 기능 검증 (호스트 + 디바이스)
- **호스트 검증**: ✅ 완료 (2026-03-27)
  - Qwen 2.5 1.5B, 1120 토큰 프롬프트 (NIAH N-PASS)
  - chunk=0 (baseline): Prefill 123.7 tok/s, TTFT 9244 ms, Decode 24.4 tok/s
  - chunk=512 (chunked): Prefill 129.6 tok/s, TTFT 8743 ms, Decode 25.1 tok/s
  - NIAH passkey 58291 양쪽 정확 추출 ✅
  - `[Prefill] Chunked mode: 1120 tokens in chunks of 512` 로그 확인 ✅
  - 성능 저하 없음 (오히려 chunked가 약간 빠름)
- **Status**: DONE
- **Sprint**: current
- **Dependencies**: Rust Developer RESIL-3 완료 ✅
- **Description**:
  **A. 호스트 검증**:
  - Qwen 2.5 1.5B, 3K+ token prompt, `--prefill-chunk-size 512`
  - 기대: prefill 정상 완료, sampling 결과가 `--prefill-chunk-size 0`과 동일
  - 피크 RSS 비교: chunk=0 vs chunk=512 → ~1.9 GB 차이 확인

  **B. S25 검증**:
  - 3401-token prompt + `--prefill-chunk-size 512`
  - 기대: OOM 없이 prefill 완료 (기존: logits 1.95 GB로 OOM/reboot)

  **C. (선택) Gemma 2 2B 동일 테스트**:
  - vocab=256000이므로 기존 logits = 3.24 GB → chunked로 ~1 MB
  - 더 극적인 RSS 절감 확인
- **디바이스 검증**: ✅ 완료 (2026-03-27, S25)
  - chunk=0: Prefill 80.1 tok/s, TTFT 14342 ms, Decode 12.8 tok/s
  - chunk=512: Prefill 58.1 tok/s, TTFT 19471 ms, Decode 12.3 tok/s
  - NIAH passkey 58291 양쪽 정확 추출 ✅
  - Prefill 속도 ~27% 저하 (chunk 간 동기화), Decode 동등
- **Acceptance Criteria**:
  - chunk=512에서 prefill 정상 완료 ✅
  - NIAH passkey 동일성 확인 ✅

## [P0] RESIL-T3. Session 9 재현 — Mode A vs C 생존율 분리
- **Status**: IN_PROGRESS
- **Sprint**: current
- **Dependencies**: RESIL-T1 ✅, RESIL-T2 ✅
- **파이프라인 검증** (2026-03-27): ✅ 완료
  - Manager TCP 리스닝 + Engine 연결 성공 (`Client connected from 127.0.0.1:49680`)
  - Manager가 Thermal Warning/Critical, Compute Warning 감지 → Directive 전송 (seq=1,2,3...)
  - Mode A baseline: Prefill 82.4 tok/s, Decode 10.2 tok/s (512 tokens, no OOM)
  - Mode C baseline: Prefill 77.0 tok/s, Decode 7.8 tok/s (512 tokens, no OOM, directives received)
- **Pressure 실험**: 부분 진행
  - 3개 generate 인스턴스 동시 실행 → MemAvail 573 MB (total ~9GB 사용)
  - Mode A가 모델 로딩 중 35초 후 종료 (OOM 또는 LMK kill)
  - 디바이스 USB 연결 끊김 (LMK/reboot)
  - **→ 전용 pressure injection tool 필요**
- **memfill tool** (tools/memfill.c): 구현 완료
  - mmap + non-compressible 데이터 + 지속 re-dirty로 zram 우회
  - 문제: S25의 zram 8GB가 할당된 메모리를 압축 → mlock은 root 권한 필요
  - 해결 방향: (A) 점진적 pressure (1GB씩 증가) (B) adb root 권한 확보 (C) 앱 전환 기반 pressure
- **남은 작업**:
  - [ ] 디바이스 재연결 후 개선된 memfill 배포 및 테스트
  - [ ] 안정적 pressure 수준 캘리브레이션 (OOM 직전 상태 유지)
  - [ ] Mode A 3회 반복
  - [ ] Mode C 3회 반복
  - [ ] 결과 테이블 작성
- **Acceptance Criteria**:
  - Mode C 생존 시간 > Mode A × 3 (3회 중 최소 2회)
  - Mode C에서 madvise RSS 감소 로그 관측
  - 결과 테이블 (3+ runs × 2 modes)

## [P1] RESIL-T4. Mode B 검증 — Offload + Resilience
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Rust Developer RESIL-6 완료 + RESIL-T3 통과
- **Description**: Mode B (lossless resilience: KV offload to disk) 동작 검증 (S25).
  ```bash
  llm_manager &
  generate --model-path /data/local/tmp/models/qwen2.5-1.5b \
      --kv-offload disk --enable-resilience \
      --resilience-transport tcp:127.0.0.1:9876 \
      --prefill-chunk-size 512 \
      --prompt "..." -n 512
  ```
  - manager.log에 "client connected" 확인
  - gen.log에 directive 수신 + offload 실행 확인
  - Pressure 시 KV가 disk로 offload, 회복 시 reload 확인
- **Acceptance Criteria**:
  - TCP 연결 성공 (기존 버그 해소)
  - Warning pressure에서 KV offload 동작 확인
  - Mode A/B/C 3-tier 비교 결과 수집 (논문 Figure용)

## [P1] RESIL-T5. Gemma 2 2B KV 비중 차이 실험
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: RESIL-T3 통과 + Gemma 2 2B 모델 준비
- **Description**: Gemma 2 2B는 KV slot이 Qwen의 ~2배 (2048 vs 1024 bytes/slot/layer).
  동일 시퀀스 길이에서 KV 비중이 더 높아 eviction 효과 극대화 기대.
  - 조건: 3401-token prompt, 동일 pressure, Mode A vs C
  - Qwen 2.5 1.5B 결과와 비교
  - 기대: Gemma에서 Mode A/C 분리가 더 극적
- **Acceptance Criteria**:
  - Gemma 2 2B Mode C 생존 마진 > Qwen Mode C 생존 마진
  - 결과 비교 테이블 (Qwen vs Gemma × Mode A/C)
- **Notes**: 논문에서 "모델별 KV 비중에 따른 resilience 효과 차이"를 보여줄 수 있음

---

## [P1] ZramStore Blosc 필터 실험 검증
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Rust Developer의 bytedelta/trunc_prec 구현 완료
- **Description**: |
  Blosc 필터 적용 후 실제 모델 데이터에서의 압축률과 속도 변화를 검증한다.

  ### 검증 항목

  **A. 정확성 검증**
  - bytedelta encode → decode roundtrip bit-exact
  - trunc_prec(N) 적용 후 마스킹된 비트가 정확히 0인지 확인
  - 전체 파이프라인 roundtrip: 원본 → trunc → shuffle → bytedelta → LZ4/Zstd → 역순 → 복원
  - 무손실 모드(trunc_bits=0): 원본과 bit-exact 동일
  - 손실 모드(trunc_bits>0): 마스킹된 비트만 차이, 나머지 동일

  **B. 압축률 검증 (실측)**
  - 테스트 데이터: Llama 3.2 1B 실제 추론에서 추출한 F16 KV 캐시
  - 5개 파이프라인 조합 × 16 레이어 = 80개 측정점
  - 각 측정점: 압축률, 압축 시간, 해제 시간
  - 합격 기준:
    - 무손실(bytedelta+LZ4): 압축률 > 1.05x (1.0x보다 나아야 의미 있음)
    - 무손실(bytedelta+Zstd): 압축률 > 1.1x
    - 손실(trunc5+bytedelta+Zstd): 압축률 > 1.5x
  - 해제 속도: 4MB 블록 기준 < 3ms (ARM64 예산)

  **C. 추론 품질 검증 (trunc_prec)**
  - 동일 프롬프트로 BASE vs trunc(3) vs trunc(5) 생성 비교
  - greedy decoding으로 토큰 레벨 일치율 측정
  - trunc(3): 95% 이상 토큰 일치 기대
  - trunc(5): 80% 이상 토큰 일치 기대 (허용 범위)

- **Acceptance Criteria**:
  1. 정확성 테스트 전체 통과
  2. 5개 조합의 벤치마크 결과 테이블 (압축률 / 압축 ms / 해제 ms)
  3. 레이어별 압축률 분포 차트 (16 layers × 5 configs)
  4. trunc_prec 품질 비교 리포트
  5. 최종 권장 설정 판정 근거
- **Notes**: |
  - 기존 Phase 3 벤치마크 방법론 재활용 (`experiments/reports/kv_offload_phase3_perf_report.md`)
  - 합성 데이터 vs 실제 모델 데이터 차이 반드시 구분 (기존 1.0x 문제의 원인)
  - 결과 리포트: `experiments/reports/blosc_filter_experiment_report.md`
