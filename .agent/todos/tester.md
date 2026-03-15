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
