# Multi-Domain Policy Selection: MOMAB Indicator-weighted Scalarization

> **작성일**: 2026-04-15
> **대상**: `manager/scripts/policy_default.lua`의 도메인 선정 및 액션 argmax 로직을 multi-objective 관점으로 재설계
> **상태**: **설계안 (미구현)** — 본 문서는 `policy_default.lua` v1.0.1의 단일 max-domain argmax를 MOMAB 기반 indicator-weighted scalarization으로 교체하는 방안을 기술한다. 구현은 별도 작업으로 수행한다.
> **관련**: `docs/43_production_lua_policy_design.md`, `docs/42_policy_simulator_guide.md`, `manager/src/lua_policy.rs`, `manager/scripts/policy_default.lua`
> **Spec ID 후보**: MGR-POL-2xx 대역 (미할당, 본 문서 채택 시 `/spec-manage`로 할당)

---

## 1. 문제 정의

### 1.1 현재 방식: 단일 max-domain argmax

`policy_default.lua` v1.0.1 (2026-04-15 기준) 도메인 선정 로직 (L98–111):

```lua
local domains = {
    cpu     = p.cpu     or 0,
    gpu     = p.gpu     or 0,
    memory  = p.memory  or 0,
    thermal = p.thermal or 0,
}
local max_domain, max_val = nil, -1
for k, v in pairs(domains) do
    if v > max_val or (v == max_val and max_domain ~= nil and k < max_domain) then
        max_domain = k
        max_val    = v
    end
end
```

이후 L136–144에서 `c.relief[action][domain_key]` 하나만을 사용하여 action을 argmax한다.

이 방식은 MOMAB(Multi-Objective Multi-Armed Bandit) 용어로 **degenerate linear scalarization**이다. 즉, scalarization weight vector $w$가 max 도메인에 1, 나머지에 0을 부여하는 극단적 one-hot 형태:

$$
w_d = \begin{cases} 1 & d = \arg\max_{d'} p_{d'} \\ 0 & \text{otherwise} \end{cases}
$$

$$
\text{score}(a) = \sum_d w_d \cdot r_{a,d} = r_{a, d^*}
$$

### 1.2 실측 문제

시뮬레이터 관측 (2026-04-15):

- `memory_pressure` Emergency(≥0.90)가 222 tick 동안 지속되는 상황에서
- 동시에 `compute_guidance` Warning(cpu_pressure ∈ [0.60, 0.80)) 상태가 480 tick 지속되었음에도
- `compute_guidance`의 `cpu` 도메인 relief는 policy 결정에 **전혀 반영되지 않음**
- 결과: CPU throttle 계열 액션이 전혀 선택되지 않아 engine 프로세스 CPU 과열 완화 기회 상실

### 1.3 근본 원인 (MOMAB 관점)

- **2nd 도메인 starvation**: max 도메인이 하나라도 존재하면 나머지 도메인의 pressure는 threshold를 초과하더라도 weight=0을 부여받음
- **정보 손실**: `ctx.coef.relief[action]`는 5개 도메인(`gpu, cpu, mem, therm, lat`) 벡터를 가지고 있으나 1개 스칼라만 사용됨
- **Pareto 관점**: 현재는 Pareto front의 극단점(axis-aligned) 근처만 탐색 → convex 영역의 내부 optimal을 놓침

---

## 2. 설계 결정: Indicator-weighted Linear Scalarization

### 2.1 핵심 수식

임계값 초과 도메인들의 이진 indicator를 weight로 사용하는 linear scalarization:

$$
\text{score}(a) \;=\; \sum_{d \in D_{\text{active}}} r_{a,d} \;+\; \lambda \cdot \min(r_{a,\text{lat}}, 0)
$$

여기서:

- $D_{\text{active}} = \{\, d \in \{\text{gpu, cpu, mem, therm}\} \;:\; p_d \ge \theta_d \,\}$
- $\theta_d$: 도메인별 활성화 임계값 (아래 §3.2 참조)
- $r_{a,d}$: `ctx.coef.relief[action][d]` (양수=개선)
- $\lambda = 2.0$: latency penalty 계수 (§2.5 참조)

$D_{\text{active}}$가 비어 있으면 (정의상 trigger도 없음) 개입하지 않는다.

### 2.2 MOMAB 분류

본 설계는 **indicator-weighted linear scalarization** (Drugan & Nowé 2013, §III-B)에 해당한다:

- `scalarized UCB1` 계열 중 weight가 시간에 따라 변화하는 **contextual MOMAB** (Tekin & Turğay 2018)
- Weight는 이산 이진값 {0, 1}이므로 "dominant objective detection" 기법의 soft 버전이라 볼 수 있음
- **한계**: linear scalarization은 Pareto front의 **convex 영역만 커버**한다. concave 영역의 optimal action은 어떤 weight 조합으로도 선택되지 않는다 (Roijers et al. 2013, §4.1). 이 한계는 Chebyshev scalarization으로 극복 가능 (§5 trade-off 표 참조).

### 2.3 Emergency Weighted Gate

**문제**: 어떤 도메인이 Emergency 수준이면 해당 도메인을 최우선 처리해야 하지만, pure lexicographic 순서를 강제하면 Emergency가 지속되는 동안 Critical 수준의 다른 도메인이 **영원히 무시되는 starvation**이 발생한다.

**해결**: weight 3:1 혼합 방식

$$
w_d = \begin{cases}
3.0 & p_d \ge \theta_d^{\text{emergency}} \\
1.0 & \theta_d^{\text{critical}} \le p_d < \theta_d^{\text{emergency}} \\
1.0 & \theta_d^{\text{warning}}  \le p_d < \theta_d^{\text{critical}} \\
0.0 & p_d < \theta_d^{\text{warning}}
\end{cases}
$$

$$
\text{score}(a) \;=\; \sum_{d} w_d \cdot r_{a,d} \;+\; \lambda \cdot \min(r_{a,\text{lat}}, 0)
$$

**이론적 근거**: Wray et al. (2015) "Lexicographic MDP"에서 상위 objective가 극단적 상황에서만 활성화되는 경우 pure lexicographic 대신 "slack-based relaxation"을 권장. 3:1 ratio는 Emergency 도메인의 relief가 다른 Critical 도메인 relief의 1/3 이하로 작을 때만 non-Emergency 도메인이 선택되도록 보장한다.

**예시**:
- memory Emergency, cpu Critical 동시 발생
- action A: `r_mem=0.4, r_cpu=0.0` → score = 3.0·0.4 = 1.2
- action B: `r_mem=0.0, r_cpu=0.5` → score = 1.0·0.5 = 0.5
- action C: `r_mem=0.3, r_cpu=0.5` → score = 3.0·0.3 + 1.0·0.5 = 1.4 ← 선택

### 2.4 Latency 처리: Hard Floor + Soft Penalty

기존 방식 (v1.0.1): `lat >= -0.20`이면 후보, 아니면 배제 (hard cutoff).

**문제**: `-0.20`과 `-0.21` 경계에서 discontinuity 및 information loss.

**새 방식**:

1. **Hard floor**: $r_{a,\text{lat}} < -0.30$ 인 action은 후보 제외
   - **예외**: $D_{\text{active}}$에 Emergency 도메인이 포함된 경우 hard floor를 완화 (`-0.50`까지 허용)
   - 사유: Emergency 상황에서는 latency 손실을 감수해서라도 압박 완화가 우선
2. **Soft penalty**: 후보 내에서 $\lambda \cdot \min(r_{a,\text{lat}}, 0)$ 항 추가
   - $\lambda = 2.0$: resource domain(weight 1.0)의 2배
   - 사유: latency는 user-facing metric이며 resource pressure는 intermediate metric. User QoS 기준으로 보면 latency 악화의 비용이 resource 완화의 이익보다 크다.

### 2.5 Hysteresis (임계값 근처 플리커 방지)

**문제**: $p_d$가 $\theta_d$ 근처에서 진동하면 도메인이 매 tick ON/OFF로 토글되어 policy가 불안정.

**해결**: 이력 기반 hysteresis — 한 번 active된 도메인은 $0.85 \cdot \theta_d$ 아래로 내려갈 때까지 active 유지.

$$
d \in D_{\text{active}}^{t}
\iff
\begin{cases}
p_d^{t} \ge \theta_d & \text{if } d \notin D_{\text{active}}^{t-1} \\
p_d^{t} \ge 0.85 \cdot \theta_d & \text{if } d \in D_{\text{active}}^{t-1}
\end{cases}
$$

**이전 tick의 $D_{\text{active}}^{t-1}$ 추론**: `ctx.history`의 마지막 항목에서 각 도메인의 `p` 값과 이진 active 여부를 역산. `ctx.history`는 현재 Lua에서 읽기만 가능하며 별도 저장 계층은 필요하지 않다.

**한계**: `ctx.history` 유실 시(최초 tick, 재시작 직후) hysteresis는 no-op로 동작 (즉, 일반 threshold만 사용). 이는 허용 가능한 degradation.

---

## 3. 구현 상세

### 3.1 Lua 구현 스케치

```lua
-- 도메인별 임계값 (Rust MemoryMonitorConfig::default() 및 ComputeMonitor defaults에 정렬)
-- NOTE: dual source of truth — §3.2 참조
local THRESHOLD = {
    mem   = { warning = 0.60, critical = 0.80, emergency = 0.90 },
    cpu   = { warning = 0.60, critical = 0.80, emergency = 0.90 },
    gpu   = { warning = 0.60, critical = 0.80, emergency = 0.90 },
    therm = { warning = 0.70, critical = 0.85, emergency = 0.95 },
}
local HYSTERESIS_FACTOR = 0.85
local EMERGENCY_WEIGHT  = 3.0
local LATENCY_LAMBDA    = 2.0
local LATENCY_FLOOR     = -0.30
local LATENCY_FLOOR_EMERGENCY = -0.50

-- 도메인 이름 매핑: pressure → relief 키
local DOMAIN_KEY = { memory = "mem", thermal = "therm", cpu = "cpu", gpu = "gpu" }

local function prev_active(history)
    -- ctx.history 마지막 entry에서 이전 tick의 active 도메인 이진 set 추론
    -- 없으면 빈 테이블 반환 → hysteresis no-op
    local set = {}
    if not history or #history == 0 then return set end
    local last = history[#history]
    for _, d in ipairs({"mem", "cpu", "gpu", "therm"}) do
        if last[d .. "_active"] then set[d] = true end
    end
    return set
end

local function domain_weights(pressure, history)
    local prev = prev_active(history)
    local w = {}
    for pkey, rkey in pairs(DOMAIN_KEY) do
        local p   = pressure[pkey] or 0
        local thr = THRESHOLD[rkey]
        local was_active = prev[rkey] == true
        local floor_val = was_active and (thr.warning * HYSTERESIS_FACTOR) or thr.warning

        if p >= thr.emergency then
            w[rkey] = EMERGENCY_WEIGHT
        elseif p >= floor_val then
            w[rkey] = 1.0
        else
            w[rkey] = 0.0
        end
    end
    return w
end

local function scalarize(relief, weights, has_emergency)
    local s = 0
    for dkey, w in pairs(weights) do
        if w > 0 then
            s = s + w * (relief[dkey] or 0)
        end
    end
    local lat = relief.lat or 0
    if lat < 0 then
        s = s + LATENCY_LAMBDA * lat
    end
    return s
end

-- decide(ctx) 본문 일부
local w = domain_weights(p, ctx.history)
local has_emergency = false
for _, wv in pairs(w) do
    if wv >= EMERGENCY_WEIGHT then has_emergency = true; break end
end

if not has_emergency and next(w) == nil then
    return {}  -- D_active = empty
end

local lat_floor = has_emergency and LATENCY_FLOOR_EMERGENCY or LATENCY_FLOOR

local best_action, best_score = nil, -math.huge
for action, r in pairs(c.relief) do
    if (r.lat or 0) >= lat_floor then
        local s = scalarize(r, w, has_emergency)
        if s > best_score then
            best_action, best_score = action, s
        end
    end
end
```

### 3.2 Threshold Dual Source of Truth (미해결)

**현재 상태**: threshold 값이 두 곳에 **독립적으로 하드코딩**되어 있다.

1. Rust: `manager/src/monitor/memory.rs::MemoryMonitorConfig::default()` — `available_pct` 기준 (warning=40%, critical=20%, emergency=10%)
2. Lua: 본 스크립트 `THRESHOLD` 테이블 — `used_pct` 기준 (0.60, 0.80, 0.90)

**위험**: Rust 쪽 config를 변경하면 Lua 쪽이 silent drift → 도메인 활성화 기준 불일치.

**향후 개선안** (TODO, 미구현):

- `ctx.coef.threshold` 필드를 `build_ctx()`에 추가하여 Rust의 monitor config를 Lua에 주입
- 구조:
  ```
  ctx.coef.threshold
      .mem.{warning, critical, emergency}     : f32
      .cpu.{warning, critical, emergency}     : f32
      .gpu.{warning, critical, emergency}     : f32
      .therm.{warning, critical, emergency}   : f32
  ```
- 예상 변경 범위: `manager/src/lua_policy.rs::build_ctx()`, `manager/src/monitor/{memory,compute,thermal}.rs` 의 config getter 노출 — 약 50줄
- Spec ID 할당 필요 (MGR-POL-2xx 대역)

**현재 workaround**: Lua `THRESHOLD` 테이블 상단 주석에 "Rust와 정렬 유지 필요" 명시 및 changelog로 drift 추적.

### 3.3 Hysteresis 상태 영속성 (미해결)

**현재 방식**: `ctx.history`의 마지막 entry에서 이진 active 여부 역산.

**제약**:

- `ctx.history` entry 포맷이 도메인별 `{d}_active : bool` 필드를 포함한다고 가정 — 현재 `lua_policy.rs::build_ctx()`는 이를 **기록하지 않음**
- 따라서 현재 설계는 `build_ctx()` 내 history push 로직 확장이 필요 (약 20줄)
- 대안: Rust 쪽에서 `HashMap<Domain, bool>` active set을 PolicyState로 유지하고 `ctx.prev_active : string[]`로 주입 — 이 경우 Rust 변경이지만 Lua는 단순 읽기

**결정 보류**: 두 방식의 테스트 가능성 비교 후 선택. 본 문서에서는 단일 접근을 고정하지 않는다.

---

## 4. Trade-off

| 속성 | 이진 indicator-weighted (본 설계) | 연속 pressure-weighted | Chebyshev scalarization |
|------|---|---|---|
| domain magnitude 불균등 영향 | 없음 (이진이므로 scale-free) | 있음 (memory 0.9 vs cpu 0.3 직접 합산) | 있음 (reference point 선정 필요) |
| Pareto 커버리지 | convex 영역만 | convex 영역만 | convex + concave |
| threshold 경계 discontinuity | 있음 (hysteresis로 완화) | 없음 | 없음 |
| operator 해석성 | 높음 ("임계값 넘은 도메인만 관여") | 중간 | 낮음 (utopia point 개념) |
| 파라미터 수 | $\theta_d$ per domain (3 per domain × 4 = 12) | $\theta_d$ (동일) | $\theta_d$ + reference point $z^*$ |
| 구현 복잡도 | 낮음 | 낮음 | 중간 |
| Emergency gate 통합 용이성 | 높음 (weight 3:1) | 중간 (clipping 필요) | 낮음 (norm 선택 충돌) |

**본 설계 선택 이유**:

1. **Operator 해석성**: 운영자가 Lua 스크립트를 직접 편집하는 사용 흐름에서 "threshold 넘은 도메인만 점수에 기여"는 pressure 값의 직접 가중치보다 훨씬 이해하기 쉬움.
2. **Scale 불균등 무관**: 현재 `ctx.coef.pressure`의 도메인 간 정규화 품질이 검증되지 않음 (memory와 thermal이 같은 0.0~1.0 스케일이지만 실제 분포는 다름). 이진화는 이 문제를 우회.
3. **Chebyshev는 reference point 튜닝이 추가 부담**: 운영 초기 단계에서 hyperparameter를 최소화.

**향후 재검토**: Pareto concave 영역의 optimal action이 실제로 존재하는지는 측정 후 판단. 존재한다면 `policy_chebyshev.lua` 병행 스크립트로 AB test 가능 (§6.4 참조).

---

## 5. 이론적 근거

본 설계가 참고한 논문:

- **Drugan, M. M., & Nowé, A. (2013).** *Designing Multi-Objective Multi-Armed Bandits Algorithms: A Study.* IEEE ADPRL. — MOMAB 정의, Pareto UCB1, scalarized UCB1, linear vs Chebyshev scalarization 비교. §III-B의 linear scalarization이 본 설계의 이론적 기반.
- **Roijers, D. M., Vamplew, P., Whiteson, S., & Dazeley, R. (2013).** *A Survey of Multi-Objective Sequential Decision-Making.* JAIR 48:67–113. — §4.1에서 linear scalarization이 Pareto front의 convex 영역만 커버한다는 한계를 정식화. §4.2에서 Chebyshev의 concave 영역 커버 증명.
- **Tekin, C., & Turğay, E. (2018).** *Multi-objective Contextual Multi-armed Bandit With a Dominant Objective.* IEEE Transactions on Signal Processing. — Context(본 설계의 `ctx.coef.pressure`)에 따라 dominant objective가 바뀌는 문제에 대한 regret bound 증명. §2.2의 Emergency gate는 이 논문의 "dominant objective detection"의 간소화 버전.
- **Wray, K. H., Zilberstein, S., & Mouaddib, A.-I. (2015).** *Multi-Objective MDPs with Conditional Lexicographic Reward Preferences.* AAAI. — Lexicographic preference를 conditional (i.e., 상위 objective가 threshold를 넘을 때만 활성) 로 relaxation하는 방식. §2.3 Emergency weighted gate의 이론 근거.
- **Pike-Burke, C., Agrawal, S., Szepesvari, C., & Grunewalder, S. (2018).** *Bandits with Delayed, Aggregated Anonymous Feedback.* ICML. — 본 프로젝트의 3초 heartbeat 지연에 따른 relief 관측이 delayed anonymous feedback에 해당. EWMA relief 학습의 bias 분석에 참고.

---

## 6. 미결 사항 (향후 검토)

### 6.1 Threshold 주입 인프라

- **범위**: Rust `MemoryMonitorConfig`, `ComputeMonitorConfig`, `ThermalMonitorConfig` 의 threshold 값을 `ctx.coef.threshold`로 Lua에 노출
- **변경량**: `manager/src/lua_policy.rs::build_ctx()` + 각 monitor config getter 추가 = 약 50줄
- **Spec**: MGR-POL-2xx 대역에서 ID 할당 후 `spec/` 및 `arch/` 반영
- **우선순위**: 중 — drift가 실제 문제가 되기 전까지는 주석으로 완화

### 6.2 Hysteresis 상태 영속성

- Lua `ctx.history` 확장 vs Rust PolicyState 관리 중 선택
- 테스트 가능성: Rust 관리 쪽이 unit test 작성 용이
- 결정 보류, 구현 단계에서 결정

### 6.3 Action 계열 반복 방지 (Q2)

- `.agent/todos/backlog.md`의 기록 항목과 연관
- 현재 `is_active()` 체크로 직전 tick 중복만 방지 — scalarization 전환 후에는 여러 도메인의 relief가 합산되어 다른 action이 선택될 가능성이 높아지므로 순환 자체가 감소할 것으로 예상
- 실측 필요

### 6.4 Chebyshev scalarization AB test

- `policy_chebyshev.lua` 병행 스크립트:
  $$
  \text{score}(a) = -\max_{d \in D_{\text{active}}} \left[ w_d \cdot (z_d^* - r_{a,d}) \right]
  $$
- Reference point $z^* = [\max_a r_{a,d}]$ (tick마다 재계산)
- 시뮬레이터에서 동일 trace 재생하여 본 설계와 경로 비교
- 판단 기준: Pareto concave 영역에서 본 설계가 놓치는 action이 유의미한 빈도로 존재하는가

---

## 7. 구현 작업 목록 (Implementer 위임용)

본 문서 채택 시 아래 작업을 Implementer에게 위임:

1. `manager/scripts/policy_default.lua` 로직 교체 (v2.0.0)
2. Changelog 갱신: 버전 bump + 2.0.0 노트 추가
3. `docs/42_policy_simulator_guide.md` 의 policy behavior 섹션 갱신
4. 시뮬레이터 trace로 regression test: memory Emergency + compute Warning 동시 발생 시 cpu 도메인 액션이 선택되는지 확인
5. (선택) `policy_chebyshev.lua` 병행 스크립트 신규 작성 — §6.4 AB test용

**구현 시 참고**: 본 문서의 §3.1 스케치 코드는 **완전하지 않음** (예: `build_cmd` 호출, active 중복 가드, Emergency 보조 throttle 등 기존 v1.0.1의 나머지 로직은 보존). 구현자는 교체 범위를 "도메인 선정 + action argmax" 두 블록으로 한정할 것.
