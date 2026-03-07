# Rust Developer TODO

> **역할**: 백엔드 구현, 모델 추론 로직, 성능 최적화, 버그 수정
> **소유 영역**: `engine/src/`, `manager/`, `shared/`

---

## [P0] generate.rs Experiment Mode 구현
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음
- **Description**: generate.rs에 실험 모드 추가. 토큰 위치 기반 신호 주입 + per-token JSONL 출력 + 시스템 메트릭 수집
- **Acceptance Criteria**:
  - CLI 플래그: `--experiment-schedule`, `--experiment-output`, `--experiment-logits-topk`, `--experiment-sample-interval`, `--greedy`
  - 스케줄 JSON 파싱: `{"name":...,"signals":[{"at_token":N,"signal":{...}}]}`
  - 내부 mpsc 채널로 ResilienceManager 생성 (외부 transport 불필요)
  - 토큰 위치 매칭 시 신호 주입 → 기존 resilience checkpoint 흐름 활용
  - per-token JSONL 출력: pos, token_id, text, tbt_ms, forward_ms, signal, actions, cache_pos, throttle_ms, top_logits
  - `_summary` 레코드 출력 (마지막 줄)
  - `--greedy` 플래그로 temperature=0 강제
  - `cargo test` 통과, `sanity_check.sh` 통과
- **Notes**: `experiments/PLAN.md` Section 3 참조. **전체 실험의 선행 조건 — 최우선 구현**

## [P0] SystemSampler 구현
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Experiment Mode 구현과 동시 진행
- **Description**: per-N-token 시스템 메트릭 수집기. `--experiment-sample-interval N`으로 빈도 제어
- **Acceptance Criteria**:
  - Process RSS: `/proc/self/statm` 읽기
  - CPU Utilization: `/proc/self/stat` delta 계산
  - CPU Frequency: `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq` 코어별 읽기
  - Thermal: `/sys/class/thermal/thermal_zone*/temp` zone별 읽기
  - GPU Freq/Util: Android sysfs 경로 (없으면 null)
  - interval=0이면 비활성화, interval=N이면 N 토큰마다 수집
  - JSONL의 `sys` 필드로 출력, 비수집 토큰은 `sys:null`
  - `_summary`에 sys_start/sys_end + governor 정보 포함
- **Notes**: `experiments/PLAN.md` Section 3.4 참조. 호스트 x86_64 기준 interval=1에서 오버헤드 < 0.5% 목표

## [P0] Experiment 스케줄 JSON 파일 작성
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Experiment Mode 구현 완료
- **Description**: Round 1~2 실험용 스케줄 JSON 파일 생성. `experiments/configs/` 디렉토리
- **Acceptance Criteria**:
  - baseline.json (빈 스케줄)
  - 속도 실험 (128 토큰): thermal_warning_32, thermal_critical_32, thermal_crit_32_recover_96, compute_cpu_32, energy_emergency_32
  - 품질 실험 (512 토큰): memory_warning_256, memory_critical_256, memory_crit_256_recover_384
  - 메모리 실험 (1024 토큰): memory_critical_512, memory_critical_256_long, memory_critical_768, memory_crit_512_recover_768
  - 모든 JSON이 스케줄 스키마 준수
- **Notes**: `experiments/PLAN.md` Section 3.2 + Section 5 참조

---

## Archive (완료)

<details>
<summary>DONE 항목 (접기)</summary>

## [P1] IPC Transport 추상화 구현
- **Status**: DONE
- **Notes**: 커밋 c2b7c64. SignalTransport trait + DbusTransport + UnixSocketTransport + MockTransport

## [P1] Hybrid + Eviction 통합 구현
- **Status**: DONE
- **Notes**: 커밋 dc78418. CacheManager, EvictionPolicy(none/sliding/h2o) 통합

## [P1] Hybrid + Resilience 통합 구현
- **Status**: DONE
- **Notes**: 커밋 db2df27. SwitchBackend 실제 동작, GPU↔CPU 양방향 전환

## [P2] Manager 서비스 Collector 구현
- **Status**: DONE
- **Notes**: 커밋 bd57980. 4개 Collector, 13개 테스트

## [P2] Manager 서비스 PolicyEngine + Emitter 구현
- **Status**: DONE
- **Notes**: 커밋 89e2a34 + b35803a. OCP PolicyEngine trait, 26개 테스트

</details>
