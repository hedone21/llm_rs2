# 23. Resilience 통합 테스트 전략

> **상태**: 설계 완료
> **대상**: Resilience Manager + generate.rs 통합 검증
> **구현 참조**: `docs/22_resilience_integration.md`

---

## 1. 테스트 계층

| 계층 | 환경 | 범위 | 실행 방법 |
|------|------|------|-----------|
| **L1 — Unit** | 호스트 (x86/ARM) | 개별 모듈 격리 | `cargo test` |
| **L2 — Integration** | 호스트 | 모듈 간 상호작용 (mock 기반) | `cargo test --features resilience --test test_resilience_integration` |
| **L3 — E2E** | Android 디바이스 | 실제 D-Bus + 추론 루프 | `stress_test_adb.py` |

---

## 2. L1 — 유닛 테스트 (구현 완료)

이미 각 모듈 내 `#[cfg(test)]`에 구현되어 있음.

| 모듈 | 파일 | 테스트 수 | 검증 내용 |
|------|------|-----------|-----------|
| ResilienceManager | `src/resilience/manager.rs` | 8 | poll, 상태 전이, 채널 끊김, 액션 실행 |
| Strategy (4종) | `src/resilience/strategy/*.rs` | 각 2~3 | 레벨별 반응, 액션 생성 |
| resolve_conflicts | `src/resilience/strategy/mod.rs` | 7 | Suspend 우선, min evict, max delay |
| Signal/Level | `src/resilience/signal.rs` | 7 | 파싱, 정렬, 추출 |
| OperatingMode | `src/resilience/state.rs` | 5 | 레벨 조합 → 모드 결정 |

---

## 3. L2 — 통합 테스트 시나리오 (구현 완료)

파일: `tests/test_resilience_integration.rs`

### 시나리오 목록

| # | 시나리오 | 입력 | 예상 결과 | 검증 방법 |
|---|----------|------|-----------|-----------|
| 1 | **Eviction Flow** | MemoryPressure(Critical) | Evict 액션 생성, target_ratio ∈ (0, 1) | `actions.iter().any(Evict)`, KV position 감소 시뮬레이션 |
| 2 | **Throttle Flow** | ThermalAlert(Critical, ratio=0.5) | Throttle 액션, delay > 0 | `ctx.throttle_delay_ms > 0` |
| 3 | **Suspend Flow** | EnergyConstraint(Emergency) | Suspend 액션, mode=Suspended | `ctx.suspended == true`, `mgr.mode() == Suspended` |
| 4 | **Disabled Noop** | resilience_manager = None | 상태 변화 없음 | `throttle_delay_ms == 0` |
| 5 | **Restore Defaults** | Critical → Normal 전이 | RestoreDefaults 실행, 제약 해제 | `throttle == 0, reject == false` |
| 6 | **Channel Disconnect** | 송신자 drop | 패닉 없음, 상태 유지 | `actions.is_empty()`, mode 유지 |
| 7 | **LimitTokens Min** | LimitTokens(100) → LimitTokens(300) | min(100, 300) = 100 | `num_tokens == 100` |

### 추가 설계 (미구현, 향후 확장)

| # | 시나리오 | 설명 | 우선순위 |
|---|----------|------|----------|
| 8 | **다중 신호 동시 수신** | Memory(Warning) + Thermal(Critical) 동시 → resolve_conflicts 검증 | P2 |
| 9 | **Eviction + Throttle 복합** | Critical memory + Critical thermal → Evict + Throttle 동시 발생 | P2 |
| 10 | **급속 상태 전이** | Normal → Emergency → Normal (빠른 회복) | P2 |
| 11 | **대량 신호 누적** | 100개 신호 버퍼링 후 poll → 성능 저하 없음 | P3 |

---

## 4. L3 — E2E 디바이스 테스트 시나리오

> **전제**: Android 디바이스에 D-Bus `org.llm.Manager1` 서비스 실행 중

### 4.1 Mock Manager를 이용한 E2E 테스트

```bash
# 터미널 1: Mock Manager 실행
adb shell /data/local/tmp/mock_manager

# 터미널 2: Resilience 활성화 추론
adb shell /data/local/tmp/generate \
  --model-path /data/local/tmp/model \
  --backend cpu -n 500 \
  --enable-resilience
```

### 4.2 디바이스 테스트 시나리오

| # | 시나리오 | Mock Manager 동작 | 검증 (로그 파싱) |
|---|----------|-------------------|-----------------|
| D1 | **정상 추론 (Resilience 비활성)** | 없음 | 추론 정상 완료, `[Resilience]` 로그 없음 |
| D2 | **정상 추론 (Resilience 활성)** | 신호 없음 | `[Resilience] Manager enabled` 출력, 추론 정상 완료 |
| D3 | **메모리 압박 Eviction** | 30초 후 MemoryPressure(Critical) | `[Resilience] Evicted N tokens` 로그, 추론 계속 |
| D4 | **온도 Throttle** | 20초 후 ThermalAlert(Critical) | TBT 증가 확인 (throttle 효과) |
| D5 | **배터리 Suspend** | 45초 후 EnergyConstraint(Emergency) | `[Resilience] Inference suspended` 로그, 추론 중단 |
| D6 | **D-Bus 연결 실패** | Manager 미실행 | `D-Bus listener exited` 경고, 추론 정상 완료 |
| D7 | **신호 → 복구 주기** | 30초 Critical → 60초 Normal → 90초 Critical | 복구 후 정상 동작, 반복 eviction 안정성 |
| D8 | **장시간 안정성** | 5분마다 랜덤 신호 | 1시간 연속 추론, 메모리 누수 없음 |

### 4.3 검증 기준

| 항목 | 기준 |
|------|------|
| 추론 안정성 | 비정상 종료 0건 (Suspend 제외) |
| 메모리 누수 | RSS 증가율 < 1MB/hour |
| Eviction 정합성 | `current_pos` 단조 감소 후 증가 가능, `start_pos` 단조 증가 |
| Throttle 효과 | TBT 평균 > `throttle_delay_ms` |
| Fail-open | D-Bus 불가 시 추론 성능 저하 0% |

---

## 5. 테스트 자동화 방안

### 5.1 호스트 CI (GitHub Actions)

```yaml
# .github/workflows/test.yml
- name: Run unit + integration tests
  run: |
    cargo test
    cargo test --features resilience --test test_resilience_integration
```

### 5.2 디바이스 CI (stress_test_adb.py 확장)

```python
# scripts/stress_test_adb.py에 추가할 테스트 케이스
def test_resilience_eviction():
    """D3: 메모리 압박 시 eviction 동작 확인"""
    # 1. mock_manager 실행 (30초 후 MemoryPressure)
    # 2. generate --enable-resilience 실행
    # 3. stderr에서 "[Resilience] Evicted" 파싱
    # 4. 추론 완료 확인
    pass
```

---

## 6. 커버리지 매트릭스

| 모듈 | L1 | L2 | L3 |
|------|:--:|:--:|:--:|
| ResilienceManager.poll() | O | O | O |
| execute_action() | O | O | O |
| DbusListener.spawn() | - | - | O |
| resolve_conflicts() | O | O | - |
| MemoryStrategy | O | O (eviction) | O (D3) |
| ThermalStrategy | O | O (throttle) | O (D4) |
| EnergyStrategy | O | O (suspend) | O (D5) |
| ComputeStrategy | O | - | - |
| generate.rs checkpoint | - | O | O |
| Fail-open (D-Bus 없음) | O (channel) | O (noop) | O (D6) |

**현재 커버리지**: L1 완료, L2 완료 (7 시나리오), L3 설계 완료 (구현 대기)
