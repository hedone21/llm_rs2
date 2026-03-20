# Hierarchical Policy Design

## 1. 개요

이 문서는 Manager의 policy 시스템을 threshold 기반 규칙 엔진에서 hierarchical control + cross-domain action selection 으로 교체하는 설계를 기술한다.

**Supersedes**: 이 문서는 `27_manager_architecture.md`의 PolicyEngine 섹션과 `29_manager_monitor_redesign.md`를 대체한다. Monitor 계층(기존 29번 문서)은 유지하되, Monitor 이후의 정책 결정 파이프라인이 이 문서의 범위이다.

---

## 2. 현재 방식의 문제

### 2-1. 반응이 이산적이다

Level이 Normal→Warning→Critical로 점프할 때만 반응한다. CPU 69%→71%는 갑자기 Warning이 되고, 71%→89%도 여전히 같은 Warning이다. 압박의 정도를 구분하지 못한다.

### 2-2. 누적 압박을 감지하지 못한다

75%가 5초 vs 30초 유지. 후자는 지속적 경쟁이므로 더 적극적으로 대응해야 하나 현재는 이 차이를 구분하지 못한다.

### 2-3. 도메인 간 상호작용을 무시한다

현재는 compute, memory, thermal을 각각 독립 처리한다. 실제로는 `switch_hw` 하나가 compute + thermal을 동시 완화한다. 도메인별 독립 대응하면 액션을 과잉 적용하고 불필요한 품질 열화가 발생한다.

**핵심 관찰**: 하나의 액션이 여러 도메인에 동시에 영향을 미친다는 사실을 명시적으로 모델링하면, 더 적은 액션으로 더 적은 품질 손실로 같은 수준의 압력을 해소할 수 있다.

### 2-4. 품질 보장이 없다

규칙만 있고 LLM 출력 품질이 얼마나 떨어지는지에 대한 추정이나 보장이 없다.

---

## 3. 전체 구조 — 3단계 계층

```
측정값 → [PI Controller] → pressure (연속값)
              ↓
         [Supervisory Layer] → mode (이산값)
              ↓
pressure + mode + engine state → [Action Selector] → 최적 액션 조합
```

### 3-1. Level 1: PI Controller — "얼마나 심한가"

**목적**: 원시 측정값(CPU %, 온도, 메모리)을 0.0~1.0의 연속적 pressure intensity로 변환.

**왜 PI인가**: 비례항(P)은 "지금 얼마나 나쁜가", 적분항(I)은 "얼마나 오래 나빴는가". 두 가지를 합치면 일시적 spike와 지속적 부하를 자연스럽게 구분한다. 미분항(D)은 센서 노이즈에 민감하여 제외한다.

**도메인별 독립 운영**: compute, memory, thermal은 물리적으로 다른 양이고 측정 주기도 다르다(thermal은 열관성 때문에 1초, 나머지는 100ms). 각각 독립 PI로 운영한다.

**구체적 구현**:

```rust
struct PiController {
    kp: f32,
    ki: f32,
    setpoint: f32,           // 목표값 (예: CPU 0.7)
    integral: f32,
    integral_clamp: f32,     // anti-windup 상한
    can_act: bool,           // 해소 가능한 액션 존재 여부
}
```

**Anti-windup**: pressure가 지속되지만 해소 가능한 액션이 없는 상태(예: KV cache가 이미 비어서 eviction 불가)에서 적분항이 무한히 커지는 것을 방지한다. `can_act`가 false이면 적분을 동결한다.

**3개 도메인 인스턴스**:

| 도메인 | 측정값 | setpoint | 폴링 주기 |
|--------|--------|----------|-----------|
| compute | CPU utilization (0~1) | 0.70 | 100ms |
| memory | memory pressure (0~1) | 0.75 | 100ms |
| thermal | normalized temp (0~1) | 0.80 | 1000ms |

**출력**: `PressureVector { compute, memory, thermal }` — 각 0.0~1.0

**Cold start**: PI의 적분항이 충분히 누적되기 전(처음 수 초)에는 Supervisory Layer가 기존 threshold 기반 로직을 fallback으로 사용한다. PI가 안정화되면(10 폴링 사이클 후) 전환한다.

### 3-2. Level 2: Supervisory Layer — "어떤 수준의 대응이 필요한가"

**목적**: pressure vector로부터 시스템 전체의 mode(Normal / Warning / Critical)를 결정한다.

```rust
struct SupervisoryLayer {
    mode: OperatingMode,
    warning_threshold: f32,      // 0.4
    critical_threshold: f32,     // 0.7
    warning_release: f32,        // 0.25 (hysteresis)
    critical_release: f32,       // 0.50 (hysteresis)
    hold_time: Duration,         // 3~5초
    stable_since: Option<Instant>,
}
enum OperatingMode { Normal, Warning, Critical }
```

**판정 로직**:
- `peak = max(pressure.compute, pressure.memory, pressure.thermal)`
- 상승 (즉시): Normal→Warning: `peak ≥ 0.4`, Warning→Critical: `peak ≥ 0.7`
- 하강 (`hold_time` 동안 안정 유지 필요): Critical→Warning: `peak < 0.50` 유지, Warning→Normal: `peak < 0.25` 유지

**Hysteresis의 의도**: 올라갈 때는 즉시 반응(안전 우선), 내려갈 때는 일정 시간 안정적으로 유지되어야 하강한다. Mode 전환이 빈번하면 액션이 적용→해제→적용을 반복하며 떨리고, 특히 irreversible lossy 액션은 반복 적용 시 정보 영구 손실이 발생한다.

### 3-3. Level 3: Action Selector — "무엇을 할 것인가"

이하 §4, §5에서 상세 기술한다.

---

## 4. Cross-Domain Action Selector

### 4-1. 해결하려는 문제

여러 도메인에 동시에 pressure가 발생할 때, 어떤 액션 조합이 총 품질 손실을 최소화하면서 모든 pressure를 해소하는가? 이것은 multi-dimensional covering problem이다.

### 4-2. Relief Vector

모든 액션은 "어떤 도메인을 얼마나 완화하는가"를 나타내는 4D 벡터로 표현한다.

```rust
struct ReliefVector {
    compute: f32,    // 양수 = 완화
    memory: f32,
    thermal: f32,
    latency: f32,    // 음수 = 악화
}
```

**latency를 4번째 차원으로 포함하는 이유**: lossless 액션(throttle, kv_offload_disk 등)은 품질 손실은 없지만 latency를 악화시킨다. 이것을 Selector가 자동으로 판단하기 위해 relief vector에 포함한다.

**예시 테이블**:

| 액션 | compute | memory | thermal | latency |
|------|---------|--------|---------|---------|
| switch_hw (GPU→CPU) | +0.6 | 0.0 | +0.4 | -0.3 |
| throttle | +0.3 | 0.0 | +0.2 | -0.5 |
| kv_offload_disk | 0.0 | +0.7 | 0.0 | -0.6 |
| kv_evict | 0.0 | +0.8 | 0.0 | +0.1 |

이 relief vector의 값은 offline calibration에서 초기화되지만, 실제로는 Relief Estimator가 online learning으로 갱신한다(§6 참조).

### 4-3. 품질 비용(cost) — QCF 연동

각 lossy 액션의 예상 품질 열화:

```
D(action) = α × Q(action)
```

- `Q(action)`: Engine이 보고하는 QCF proxy 추정값 (on-demand, pressure > threshold 또는 mode 변경 시에만 요청)
- `α`: 오프라인 calibration 변환 계수 (`policy_config.toml`에 정의)
- Lossless 액션은 `D = 0`

**왜 proxy인가**: 실제 PPL 측정은 수백 토큰의 forward pass가 필요하여 실시간 제어에서 불가능하다. Proxy는 ordinal accuracy만 보존하면 충분하다.

### 4-4. 선택 알고리즘

**형식적 정의**:

```
minimize  Σ D(a)
subject to:
  Σ relief_d(a) ≥ p_d       — 모든 도메인의 pressure 해소
  Σ relief_latency(a) ≥ -L_max  — latency 악화 상한
  상호 배타 제약             — 예: eviction 방식은 1개만
  mode 제약                 — Warning이면 lossless만
```

Exhaustive search를 사용한다. 8개 액션 + exclusion group → 실제 조합 ~128개, μs 단위. 어떤 조합으로도 모든 pressure를 해소하지 못하면 best-effort(가능한 최대 relief 조합)를 선택한다.

### 4-5. 파라미터 결정

선택된 액션의 구체적 파라미터는 pressure 크기에 비례하여 선형 보간한다:

```
intensity 0.0 → range.max (보수적)
intensity 1.0 → range.min (공격적)
```

### 4-6. 동적 보정이 중요한 이유

같은 액션이라도 Engine 상태에 따라 효과가 다르다:

| 상황 | 영향 |
|------|------|
| KV cache가 30% 미만 | Eviction의 memory relief 대폭 감소 |
| 이미 CPU backend | switch_hw의 relief가 거의 0 |
| KV dtype이 이미 INT4 | kv_quant_dynamic 후보에서 제외 |
| Decode 초반 (token < 10) | layer skip cost 증가 |
| Prefill 단계 | SnapKV 사용 불가 |

이 보정은 Relief Estimator가 engine state를 feature로 활용하여 자동 수행한다.

---

## 5. Mode와 2단계 품질 보장

### 5-1. Warning (Lossless Guarantee)

Lossless 액션만 허용한다. Backend 전환, KV cache 디스크 이동, 토큰 생성 속도 제한 등이 해당한다. 수학적으로 출력이 동일하다(`D = 0` 보장). 단, memory pressure에 대한 해결책이 제한적이다.

### 5-2. Critical (Bounded Degradation)

Lossy 액션도 허용한다. KV cache eviction, layer skip, dynamic quantization 등이 해당한다. QCF로 사전 추정하고 총 열화를 최소화하는 조합을 선택한다. Selector가 총 `D`를 최소화하므로 불필요한 lossy 액션이 적용되지 않는다.

### 5-3. Lossy 액션의 가역성 구분

| 유형 | 예시 | 해제 가능? |
|------|------|-----------|
| Reversible lossy | layer_skip, kv_quant | O — 다음 토큰부터 중단/재양자화 가능 |
| Irreversible lossy | kv_evict, snapkv_compress | X — 삭제된 토큰 복원 불가 |

Irreversible 액션은 해제 대상이 아니라 "더 이상 적용하지 않음"만 가능하다. Registry의 `reversible` 필드로 구분한다.

---

## 6. Relief Estimator — 학습 기반 Relief 추정

### 6-1. 설계 원칙

Strategy 패턴으로 교체 가능하게 설계한다:

```rust
trait ReliefEstimator: Send + Sync {
    fn predict(&self, action: &ActionId, state: &FeatureVector) -> ReliefVector;
    fn observe(&mut self, action: &ActionId, state: &FeatureVector, actual: &ReliefVector);
    fn save(&self, path: &Path) -> io::Result<()>;
    fn load(&mut self, path: &Path) -> io::Result<()>;
    fn observation_count(&self, action: &ActionId) -> u32;
}
```

초기 구현: `OnlineLinearEstimator`. 나중에 EMA, Bayesian, GP 등으로 교체 가능하다.

### 6-2. 왜 Online Linear Regression인가

Relief 학습 문제의 특성:
- 보상이 즉시 관측됨 (액션 적용 → 수 초 내 pressure 변화)
- 장기 계획 불필요 (각 결정이 상대적으로 독립)
- 상태 공간이 작음 (연속 변수 ~13개)
- 액션 공간이 작음 (~10개)

이것은 MDP가 아니라 contextual decision problem이다. RL은 과잉이다.

**Online Linear Regression의 장점**:
- 상태의 연속성을 자연스럽게 활용 (bucket 불필요)
- 해석 가능: 계수를 보면 어떤 feature가 중요한지 파악 가능
- 샘플 효율 높음 (feature 수만큼의 관측이면 수렴 시작)
- 계산 비용 거의 0 (13×13 행렬 연산, μs 단위)

### 6-3. Feature Vector 스키마

13개 feature:

```
Index  Name                  Range     Source
─────  ────────────────────  ────────  ──────────────
0      kv_occupancy          0.0~1.0   heartbeat
1      is_gpu                0 or 1    heartbeat.backend
2      token_progress        0.0~1.0   position / max_seq
3      is_prefill            0 or 1    heartbeat.phase
4      kv_dtype_norm         0~1       dtype_bits / 32
5      tbt_ratio             0.0~∞     tbt_ema / tbt_baseline
6      tokens_generated_norm 0.0~1.0   generated / max_seq
7~12   active_action_*       0 or 1    각 주요 액션 활성 여부
```

Active-action feature로 **액션 조합의 상호작용을 학습**한다: `throttle_active=1`일 때의 eviction relief와 `throttle_active=0`일 때의 eviction relief를 자연스럽게 구분한다. 명시적 pairwise interaction term이 불필요하다.

### 6-4. 모델 구조

각 액션 `a`에 대해 독립 선형 모델:

```
relief(a) = W_a × φ(state) + b_a
```

- `W_a`: 4×13 행렬 (4개 relief 차원 × 13개 feature)
- `b_a`: 4×1 bias

Recursive Least Squares (RLS) 업데이트: forgetting factor `λ=0.995`로 최근 관측에 더 높은 가중치를 부여한다.

### 6-5. 세션 누적 학습

```
저장: {config_dir}/{device_id}_{model_id}_relief.json
내용: 각 액션별 W, bias, P matrix, observation_count
시점: 세션 종료 시 save(), 시작 시 load()
```

**Prior 초기화**: `policy_config.toml`의 기본 relief 값을 가상 관측 `N=5`로 주입한다. 실측 10~20회 쌓이면 prior 영향이 자연 감쇠한다.

---

## 7. Action Registry

Engine 등록 정보 + 오프라인 설정을 통합 관리한다:

```rust
struct ActionRegistry {
    actions: HashMap<ActionId, ActionMeta>,
    exclusion_groups: HashMap<String, Vec<ActionId>>,
}
struct ActionMeta {
    id: ActionId,
    kind: ActionKind,                  // Engine이 보고
    param_range: Option<ParamRange>,   // Engine이 보고
    alpha: f32,                        // QCF→cost 변환 계수 (설정 파일)
    reversible: bool,                  // 해제 가능 여부 (설정 파일)
}
```

**설정 파일 예시 (`policy_config.toml`)**:

```toml
[actions.switch_hw]
alpha = 0.0
reversible = true

[actions.kv_evict_sliding]
alpha = 0.12
reversible = false

[actions.layer_skip]
alpha = 0.25
reversible = true

[exclusion_groups]
eviction = ["kv_evict_sliding", "kv_evict_h2o"]
```

---

## 8. De-escalation — 복귀 시 액션 해제

Mode가 Normal로 돌아오면:

- Reversible lossy 액션 우선 해제 (품질 복구 급선무)
- Lossless 액션은 보수적으로 유지 (품질 영향 없으므로 급히 해제 불필요)
- Irreversible lossy 액션은 "더 이상 적용하지 않음"만 가능 (이미 삭제된 토큰 복원 불가)
- 초기 구현은 일괄 해제. 실험에서 해제 직후 다시 pressure가 올라가는 현상이 관찰되면 단계적 해제를 추가한다.

---

## 9. Manager Main Loop

모든 컴포넌트를 연결하는 메인 루프 의사코드:

```rust
loop {
    // ① Monitor 데이터 수집
    let readings = monitors.collect();

    // ② PI Controller → pressure vector
    let pressure = PressureVector {
        compute: pi[0].update(readings.cpu, dt),
        memory:  pi[1].update(readings.mem, dt),
        thermal: pi[2].update(readings.temp, dt),
    };

    // ③ Supervisory → mode
    let prev_mode = supervisory.mode;
    let mode = supervisory.evaluate(&pressure);

    // ④ Heartbeat 수신 (non-blocking)
    if let Some(hb) = transport.try_recv_heartbeat() {
        engine_state = hb;
    }

    // ⑤ 관측 업데이트 (이전 액션의 실측 relief)
    if let Some(ctx) = &pending_observation {
        if ctx.applied_at.elapsed() > OBSERVATION_DELAY {
            let actual_relief = ctx.pressure_before - pressure;
            for action in &ctx.applied_actions {
                estimator.observe(action, &ctx.feature_vec, &actual_relief);
            }
            pending_observation = None;
        }
    }

    // ⑥ 액션 필요 판단
    let needs_action = match mode {
        Normal => false,
        Warning | Critical => {
            mode != prev_mode || pressure.max() > last_acted_pressure * 1.2
        }
    };

    if needs_action {
        // ⑦ QCF on-demand 요청 (Critical에서 lossy 후보가 있을 때만)
        let qcf = if mode == Critical {
            let lossy_candidates = registry.lossy_candidates(&engine_state);
            if !lossy_candidates.is_empty() {
                transport.request_qcf(lossy_candidates).await
            } else { HashMap::new() }
        } else { HashMap::new() };

        // ⑧ Action Selection
        let commands = selector.select(
            &pressure, mode, &engine_state,
            &registry, &*estimator, &qcf,
            config.latency_budget,
        );

        // ⑨ Directive 전송
        if !commands.is_empty() {
            transport.send_directive(commands).await;
            pending_observation = Some(ObservationContext {
                pressure_before: pressure,
                feature_vec: build_features(&engine_state),
                applied_actions: commands.action_ids(),
                applied_at: Instant::now(),
            });
        }
    }

    // ⑩ De-escalation
    if mode == Normal && prev_mode != Normal {
        let releases = build_release_commands(&engine_state.active_actions, &registry);
        if !releases.is_empty() {
            transport.send_directive(releases).await;
        }
    }

    sleep(poll_interval).await;
}
```

---

## 10. Engine 측 변경 사항

기존 `CommandExecutor`를 확장한다:

- **StatusReporter**: heartbeat 주기 전송 (`kv_occupancy`, `backend`, `dtype`, `tbt_ema` 등)
- **QcfHandler**: on-demand QCF 계산 (기존 `engine/src/core/qcf/` 모듈 활용)
- **DirectiveHandler**: Manager directive 수신 및 실행

**역할 분리 원칙**:

Engine은 "어떻게" 실행하는 데 집중한다. 예: Manager가 `kv_evict + keep_ratio=0.5`를 전달하면, Engine이 현재 설정된 eviction 정책(Sliding/H2O)으로 실행한다.

Manager는 "무엇을 언제" 결정한다. 구체적 eviction 정책 선택은 Engine의 자율 영역이다.

---

## 11. 프레임워크 확장성

새 액션 추가 시:

1. Relief vector 정의
2. QCF 수식 정의 (lossy인 경우)
3. 상호 배타 제약 정의 (있는 경우)
4. `policy_config.toml`에 `alpha`, `reversible` 추가

Selector 알고리즘 수정이 불필요하다. 각 액션은 `(relief_vector, cost, constraints)` 튜플일 뿐이다.

---

## 12. 비교 실험에서 입증해야 하는 것

- **Cross-domain vs Domain-isolated**: 같은 pressure에서 cross-domain이 더 적은 액션으로 더 적은 품질 손실 달성
- **Warning만 vs Warning+Critical**: Critical 추가 시 OOM 방지 + bounded degradation 보장
- **llama.cpp baseline**: 아무 제어 없는 llama.cpp는 OOM kill이나 thermal throttling으로 중단

---

## 13. 설계 결정 이력

| 결정 | 선택 | 이유 |
|------|------|------|
| Selector 위치 | Manager | 논문 가시성, 시스템 수준 지식 소유, 테스트 용이성, Engine 교체 가능성 |
| Exhaustive vs Greedy | Exhaustive 기본 | 액션 ~10개, 조합 ~128, μs 단위. Greedy는 ablation용 |
| Relief 학습 | Online Linear Regression | RL 과잉, EMA bucket 부정확, 선형이면 충분 |
| 조합 효과 | Active-action feature 확장 | 별도 combinatorial 모델링 불필요 |
| Kp/Ki 튜닝 | Step response 실험 | Ziegler-Nichols는 linear plant 가정, 비선형 시스템에 부적합 |
| Relief table 위치 | TOML 설정 파일 | 플랫폼별 교체 용이 |
| Runtime 보정 | 초기 구현에서 제외 | 구현 부담 대비 논문 기여 미미, Future work |
| QCF 요청 시점 | on-demand | Normal 상태에서 오버헤드 0 |
| Latency 모델링 | ReliefVector 4번째 차원 | Lossless 부작용을 Selector가 자동 판단 |
| 세션 학습 | 세션 간 누적 | 같은 디바이스+모델이면 relief 특성 유사 |

---

## 14. 관련 문서

- `37_protocol_design.md` — Manager ↔ Engine 메시지 프로토콜 상세
- `27_manager_architecture.md` — (구) Manager 아키텍처 (이 문서로 superseded)
- `29_manager_monitor_redesign.md` — (구) Monitor 재설계 제안 (이 문서로 superseded)
- `22_resilience_integration.md` — (구) generate.rs 통합 설계
