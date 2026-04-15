# Safe-LUCB Policy Design: Amani 2019 기반 QoS-Safe Bandit

> **작성일**: 2026-04-15
> **대상**: `manager/scripts/policy_default.lua`에 Safe-LUCB(Linear UCB under safety constraints)를 적용하여 QoS safety 보장 + resource relief exploration/exploitation 균형
> **상태**: **설계안 (미구현)** — 구현은 Phase 0 (docs/44) 완료 후 Phase 1부터 시작.
> **관련**:
> - `docs/44_multi_domain_policy_design.md` (Phase 0 scalarization — foundation)
> - `docs/43_production_lua_policy_design.md`
> - `docs/42_policy_simulator_guide.md`
> - `.agent/research/2026-04-15_amani2019_mapping.md` (Researcher 분석)
> - `.agent/research/2026-04-15_policy_momab_safe_rl_shielding.md` (MOMAB/Safe-RL 조사)
> - `manager/src/lua_policy.rs::build_ctx()` (Phase 1~2 확장 대상)
> - `manager/src/relief/linear.rs::OnlineLinearEstimator` (Phase 1의 RLS estimator)
> **Spec ID 후보**: MGR-POL-3xx 대역 (미할당, 본 문서 채택 시 `/spec-manage`로 할당)

---

## 1. Overview

### 1.1 문제 정의

`policy_default.lua` v1.0.1 및 docs/44 Phase 0 scalarization은 **mean-only argmax** 구조이다:

$$
a_t = \arg\max_{a : r_{a,\text{lat}} \ge -0.30} \; \sum_{d \in D_{\text{active}}} w_d \cdot \hat{r}_{a,d}
$$

이 구조는 두 가지 문제를 가진다:

1. **Exploration 부재** — `\hat{r}_{a,d}`의 점 추정값만 본다. 관측 횟수가 적은 action(예: `kv_merge_d2o`)은 prior가 낮게 잡히면 계속 선택되지 않아 학습이 진전되지 않음 (cold-start lock-in).
2. **Safety 보장 부재** — `-0.30` hard floor는 **point estimate의 이진 gate**이다. Estimate uncertainty가 큰 상황에서 실제 cost는 $-0.45$이지만 추정치가 $-0.25$로 관측되어 safe로 잘못 분류될 수 있음.

### 1.2 목표

1. **Latency QoS safety 보장**: 높은 확률 $1 - \delta$로 실제 latency cost가 임계값 $C$를 초과하지 않도록 action을 제약
2. **Resource relief 최대화**: docs/44의 scalarized resource relief score를 최적화
3. **Exploration/exploitation 균형**: UCB bonus로 관측이 적은 action에 optimism 부여

### 1.3 접근: Amani 2019 "심장부 채택"

Amani et al. (2019) "Linear Stochastic Bandits Under Safety Constraints" (NeurIPS) 의 **assumption**은 우리 시스템과 부분적으로 일치하지 않는다 (A6 continuous convex action, A7 stationarity 등 — `.agent/research/2026-04-15_amani2019_mapping.md` §2 참조). 논문 그대로 구현하지 않고 **핵심 3개 메커니즘만 채택**한다:

1. **Ridge regression parameter estimation** (RLS, per-action)
2. **Confidence ellipsoid** $\|\phi\|_{V_a^{-1}}$ 기반 UCB bonus
3. **Pessimistic cost UB로 safe set 정의**

나머지(continuous action, stationary regret bound 등)는 우리 도메인에 맞게 수정한다 — §7 참조.

---

## 2. 수학적 프레임워크

### 2.1 표기

| 기호 | 정의 | 차원 |
|------|------|------|
| $\phi_t$ | Feature vector (현재 state) | $\mathbb{R}^D$, $D = 13$ (FEATURE_DIM) |
| $a_t$ | 선택한 action | 이산, $\|A\| = 9$ (ActionId 변형 수) |
| $y_{a, d}$ | Action $a$의 relief 차원 $d$ 관측값 | $\mathbb{R}$ |
| $\theta_{a, d}^*$ | True weight vector for $(a, d)$ | $\mathbb{R}^D$ |
| $V_{a, t}$ | Ridge matrix (per-action, cumulative) | $\mathbb{R}^{D \times D}$ |
| $\hat{\theta}_{a, d, t}$ | RLS estimate at tick $t$ | $\mathbb{R}^D$ |
| $\beta_t$ | Confidence radius (scalar) | $\mathbb{R}_+$ |
| $C$ | Latency safety constraint | `0.30` (hardcoded) |
| $D_{\text{active}}$ | docs/44 active domain set | $\subseteq \{\text{gpu, cpu, mem, therm}\}$ |

Action set은 discrete $\{a_1, \ldots, a_9\}$로 `ActionId::all()`의 9개 변형이다 — Amani §6의 convex action set과 다르게 우리는 arm별 independent ridge matrix를 관리한다.

### 2.2 Reward 정의 (docs/44와 동일)

Scalarized resource relief:

$$
\hat{r}(a, \phi) \;=\; \sum_{d} w_d \cdot \hat{r}_{a, d}(\phi) \;+\; \lambda \cdot \min(\hat{r}_{a, \text{lat}}(\phi), 0)
$$

- $w_d$: docs/44 §2.3의 `EMERGENCY_WEIGHT` / `1.0` / `0.0` indicator weight
- $\lambda = 2.0$: latency penalty 계수 (docs/44 §2.4)
- $\hat{r}_{a, d}(\phi) = \phi^\top \hat{\theta}_{a, d}$: RLS 예측 (현재 `LinearModel::predict` 로직과 동일)

**Note**: 이 reward 정의는 docs/44의 scalarization을 그대로 사용한다. Safe-LUCB는 이 reward 위에 UCB bonus와 safe set을 **더하는** 구조이다.

### 2.3 Cost 정의

$$
c(a, \phi) \;=\; -\hat{r}_{a, \text{lat}}(\phi)
$$

- 즉, latency relief가 음수(=악화)일수록 cost가 양수로 커진다
- $C = 0.30$: "latency degradation이 30% 이상이면 unsafe"로 정의
- Per-action 이므로 $\hat{r}_{a, \text{lat}}$는 action $a$의 OnlineLinearEstimator 내 latency head에서 계산

### 2.4 Confidence Radius $\beta_t$

Amani §4.1의 OFUL-style confidence radius:

$$
\beta_t \;=\; R \cdot \sqrt{2 \log(1 / \delta) + D \cdot \log\!\left(1 + \tfrac{L^2 t}{\lambda D}\right)} \;+\; B_\mu \cdot \sqrt{\lambda}
$$

파라미터:

- $R$: subgaussian noise parameter (relief 관측 노이즈 std 추정)
- $\delta$: failure probability (e.g., $0.05$)
- $L$: $\|\phi\|_2$ upper bound
- $\lambda$: ridge regularization (`LinearModel::new` 초기 P matrix의 $100.0 \cdot I$ → $\lambda = 0.01$에 해당)
- $B_\mu$: $\|\theta^*\|_2$ upper bound

**Non-stationary 보정 (Russac 2019)**:

우리 시스템은 `LinearModel::forgetting_factor = 0.995`로 discount factor $\lambda_{\text{rls}}$를 사용한다 (Amani의 $\lambda$ ridge regularization과 별개). Russac et al. (2019) "Weighted Linear Bandits for Non-Stationary Environments"의 확장:

$$
\beta_t^{\text{NS}} \;=\; R \cdot \sqrt{2 \log(1/\delta) + D \cdot \log\!\left(1 + \tfrac{L^2 \cdot \sum_{s \le t} \gamma^{2(t-s)}}{\lambda D}\right)} \;+\; B_\mu \cdot \sqrt{\lambda}
$$

$\gamma = \sqrt{\lambda_{\text{rls}}} = \sqrt{0.995} \approx 0.9975$

**현실적 튜닝**: 위 이론값은 $R, L, B_\mu$ 추정이 불확실하므로 **empirical scaling factor** $\alpha$를 config로 노출한다:

$$
\beta_t^{\text{practical}} \;=\; \alpha \cdot \beta_t^{\text{NS}}
$$

$\alpha$는 시뮬레이터 AB test로 튜닝 — §5 참조.

### 2.5 Safe Set

$$
S_t \;=\; \left\{\, a \in A \;:\; \hat{c}(a, \phi_t) + \beta_t \cdot \|\phi_t\|_{V_{a, t}^{-1}} \;\le\; C \,\right\}
$$

Equivalent form (latency 관점):

$$
S_t \;=\; \left\{\, a \in A \;:\; \hat{r}_{a, \text{lat}}(\phi_t) - \beta_t \cdot \|\phi_t\|_{V_{a, t}^{-1}} \;\ge\; -C \,\right\}
$$

즉, **latency relief의 lower confidence bound가 $-C$ 이상**인 action만 safe.

**Seed safe action**: $S_t = \emptyset$이 될 수 있음. 이 경우 **no-op**(빈 command list)을 반환하여 engine을 현재 상태로 유지한다. No-op은 정의상 안전하며 (아무것도 바꾸지 않으므로 latency 악화 없음) Amani §3의 "seed safe action" 역할을 한다.

### 2.6 Action Selection (UCB)

$$
a_t \;=\; \arg\max_{a \in S_t} \left[\, \hat{r}(a, \phi_t) + \beta_t \cdot \|\phi_t\|_{V_{a, t}^{-1}}^{\text{reward}} \,\right]
$$

$\|\phi\|_{V^{-1}}^{\text{reward}}$는 reward head의 confidence ellipsoid이다. 구조적으로 cost head와 동일한 $V_{a, t}$를 공유하므로 (LinearModel은 per-action 단일 P matrix) 같은 값이지만, **Lua에 노출할 때는 reward/cost 양쪽 다 동일한 scalar를 전달**한다 (구현 단순화).

즉, Lua 관점에서는:

- `ctx.coef.relief_ub[a][d]` = $\beta_t \cdot \|\phi_t\|_{V_{a, t}^{-1}}$ (모든 d에서 동일 값)
- `ctx.coef.cost_ub[a]` = 동일 스칼라 (latency head의 uncertainty로 해석)

### 2.7 Delayed Feedback

현재 시스템은 3초 heartbeat 지연으로 observation이 `OBSERVATION_DELAY_SECS = 3.0` 뒤에 성숙한다 (`lua_policy.rs::check_observation`). Pike-Burke et al. (2018) "Bandits with Delayed, Aggregated Anonymous Feedback"의 결과에 따르면, in-flight observation은 **아직 $V$ 행렬에 반영되지 않은 상태**이므로 $\|\phi\|_{V^{-1}}$가 실제보다 작게 추정된다.

**완화책**: in-flight observation count $n_{\text{flight}}$만큼 confidence radius에 penalty 추가:

$$
\beta_t^{\text{delayed}} \;=\; \beta_t^{\text{practical}} \cdot (1 + \kappa \cdot n_{\text{flight}})
$$

$\kappa$는 튜닝 파라미터 (초기값 $0.1$). 이론적 정당화는 Pike-Burke Thm 3.1의 delay penalty 상한 근사.

---

## 3. 구현 매핑

각 수학적 컴포넌트를 현재 Rust/Lua 파일에 매핑한다.

### 3.1 매핑 테이블

| 알고리즘 컴포넌트 | 수식 | 현재 파일/구조 | 필요 변경 | Phase |
|------|------|-------|----------|-------|
| Feature vector $\phi$ | $\mathbb{R}^{13}$ | `types.rs::FeatureVector` | — (이미 존재) | — |
| Per-action ridge matrix $V_{a, t}$ | $\mathbb{R}^{13 \times 13}$ | `relief/linear.rs::LinearModel.p_matrix` | — (이미 존재, 역 $V^{-1}$로 저장됨) | — |
| Reward mean $\hat{r}_{a, d}$ | $\phi^\top \hat{\theta}_{a, d}$ | `LinearModel.weights[d]` (dim=0,1,2) + bias | — (이미 존재) | — |
| Cost mean $\hat{c}(a)$ | $-\hat{r}_{a, \text{lat}}$ | `LinearModel.weights[3]` (latency head) | — (이미 존재) | — |
| $\beta_t$ (confidence scalar) | §2.4 수식 | 없음 | **신규**: `OnlineLinearEstimator::confidence_radius()` | Phase 1 |
| $\|\phi\|_{V^{-1}}$ (mahalanobis norm) | $\sqrt{\phi^\top V^{-1} \phi}$ | 없음 | **신규**: `LinearModel::mahalanobis_norm()` | Phase 1 |
| `predict_with_ub()` API | returns (mean, ub) | 없음 | **신규**: `ReliefEstimator` trait 확장 | Phase 1 |
| `ctx.coef.relief_ub[a][d]` (Lua) | UCB bonus per (a, d) | 없음 | **신규**: `lua_policy.rs::build_ctx()` 확장 | Phase 1 |
| `ctx.coef.cost_ub[a]` (Lua) | safe set check용 | 없음 | **신규**: `build_ctx()` 확장 | Phase 1 |
| `ctx.coef.beta_t` (Lua, diagnostic) | $\beta_t$ scalar | 없음 | **신규**: `build_ctx()` 확장 | Phase 1 |
| In-flight count $n_{\text{flight}}$ | $\|$observations queue$\|$ | `observations: VecDeque` 길이 | **신규**: `ctx.coef.inflight_count` 노출 | Phase 1 |
| Safe set $S_t$ | pessimistic cost UB gate | 없음 | **신규**: `policy_safe_lucb.lua` | Phase 2 |
| UCB action selection | $\arg\max_{a \in S_t}[\hat{r} + \beta_t \|\phi\|_{V^{-1}}]$ | 없음 | **신규**: `policy_safe_lucb.lua` | Phase 2 |
| Scaling factor $\alpha$ | $\beta_t^{\text{practical}}$ coef | 없음 | **신규**: `AdaptationConfig::safe_bandit_beta_scale` | Phase 1 |

### 3.2 M1: RLS 마이그레이션 (EwmaReliefTable → OnlineLinearEstimator)

**현재 상태의 이원화**:

- `lua_policy.rs`는 `EwmaReliefTable` 사용 (문자열 key, 6D relief, EWMA 스칼라, **P matrix 없음**)
- `relief/linear.rs::OnlineLinearEstimator`는 구현되어 있지만 **LuaPolicy에 연결되지 않음** (4D relief, per-action ridge matrix 존재)

**Safe-LUCB의 전제조건**: confidence ellipsoid 계산은 $V_{a, t}^{-1}$가 필수 → `OnlineLinearEstimator` 경로로 통일해야 함.

**M1의 범위** (Phase 1에서 수행):

1. `EwmaReliefTable`을 `OnlineLinearEstimator`로 교체 (또는 `ReliefEstimator` trait를 `LuaPolicy`에 주입)
2. 6D relief → 4D 정합성 확인 (현재 docs/44의 gpu/cpu/mem/therm/lat vs OnlineLinearEstimator의 compute/memory/thermal/latency)
3. 저장/로드 포맷 변경 (JSON 스키마) — `relief_table_path` 파일 호환성 깨짐 → migration 도구 또는 fresh start

**차원 불일치 해소**:

현재 `types.rs::ReliefVector`는 4차원(compute/memory/thermal/latency)이지만 Lua에는 6차원(gpu/cpu/mem/therm/lat/qos)이 노출된다. OnlineLinearEstimator 전환 시:

- **옵션 1**: `ReliefVector`를 6차원으로 확장 (`types.rs` 수정, Rust-side 영향 큼)
- **옵션 2**: OnlineLinearEstimator의 4차원을 유지하고 Lua에서 gpu/cpu는 compute의 sub-projection으로 처리 (현재 이미 이렇게 되어있을 가능성 — 확인 필요)

**결정 보류**: Phase 1 착수 시 Implementer가 `relief/linear.rs`의 실제 사용 여부 및 `EwmaReliefTable`과의 중복 범위를 재조사 후 결정.

### 3.3 신규 API 시그니처 (Phase 1)

```rust
// manager/src/relief/mod.rs

pub trait ReliefEstimator: Send + Sync {
    // ... 기존 메서드들 ...

    /// Safe-LUCB용: mean + UCB bonus 반환.
    /// `beta_t`는 caller가 계산하여 전달 (global scalar).
    /// Returns (mean_relief, ucb_bonus_scalar) — bonus는 모든 dim에 동일 적용.
    fn predict_with_ub(
        &self,
        action: &ActionId,
        state: &FeatureVector,
        beta_t: f32,
    ) -> (ReliefVector, f32);
}

impl OnlineLinearEstimator {
    /// Mahalanobis norm: sqrt(phi^T V^{-1} phi).
    /// P matrix는 이미 V^{-1}이므로 직접 계산 가능.
    pub fn mahalanobis_norm(&self, action: &ActionId, state: &FeatureVector) -> f32 { ... }

    /// 현재 tick의 β_t (non-stationary 보정 포함).
    /// config에서 R, δ, L, B_μ, α 파라미터를 받는다.
    pub fn confidence_radius(&self, action: &ActionId, config: &ConfidenceConfig) -> f32 { ... }
}

pub struct ConfidenceConfig {
    pub noise_std_r: f32,      // R
    pub failure_prob_delta: f32, // δ
    pub feature_norm_l: f32,    // L (‖φ‖_2 upper bound)
    pub theta_norm_b: f32,      // B_μ
    pub lambda_ridge: f32,      // λ (ridge reg)
    pub alpha_scale: f32,       // practical scaling factor
    pub kappa_delay: f32,       // delayed feedback penalty (Pike-Burke)
}
```

### 3.4 `build_ctx()` 확장 (Phase 1)

```rust
// manager/src/lua_policy.rs::build_ctx() 내부
// 기존 coef.relief 루프 확장

let beta_t = self.relief_estimator.confidence_radius(action, &self.confidence_config);
let inflight = self.observations.len() as f32;
let beta_t_delayed = beta_t * (1.0 + self.confidence_config.kappa_delay * inflight);

for action in ActionId::all() {
    let (mean_relief, ucb_bonus) =
        self.relief_estimator.predict_with_ub(action, &state, beta_t_delayed);

    // coef.relief (기존 — mean만)
    let r_entry = /* ... mean_relief ... */;
    r_tbl.set(action_to_lua_key(action), r_entry)?;

    // coef.relief_ub (신규)
    let ub_entry = lua.create_table()?;
    // 모든 dim에 동일 ucb_bonus를 설정 (Lua 쪽에서 dim별로 접근 가능하게)
    ub_entry.set("gpu",   ucb_bonus)?;
    ub_entry.set("cpu",   ucb_bonus)?;
    ub_entry.set("mem",   ucb_bonus)?;
    ub_entry.set("therm", ucb_bonus)?;
    ub_entry.set("lat",   ucb_bonus)?;
    ub_entry.set("qos",   ucb_bonus)?;
    r_ub_tbl.set(action_to_lua_key(action), ub_entry)?;

    // coef.cost_ub (신규) — safe set check용, latency 관점
    cost_ub_tbl.set(action_to_lua_key(action), ucb_bonus)?;
}
coef.set("relief_ub", r_ub_tbl)?;
coef.set("cost_ub", cost_ub_tbl)?;
coef.set("beta_t", beta_t_delayed)?;
coef.set("inflight_count", inflight)?;
```

---

## 4. Lua 정책 설계 (`policy_safe_lucb.lua`)

Phase 2에서 신규 작성한다. 기존 `policy_default.lua`(docs/44)와 병행하여 AB test 가능하도록 독립 스크립트로 분리.

### 4.1 의사코드

```lua
POLICY_META = {
    name    = "safe_lucb",
    version = "0.1.0",
}

-- Safety constraint: latency relief lower CB가 이 값 이상이어야 함
local LATENCY_SAFETY_C = 0.30

-- docs/44 scalarization 재사용
local THRESHOLD = { --[[ docs/44 §3.1 테이블 그대로 ]] }
local HYSTERESIS_FACTOR = 0.85
local EMERGENCY_WEIGHT  = 3.0
local LATENCY_LAMBDA    = 2.0

local function domain_weights(pressure, history) --[[ docs/44 §3.1 그대로 ]] end

local function compute_scalarized_reward(relief, weights)
    local s = 0
    for dkey, w in pairs(weights) do
        if w > 0 then s = s + w * (relief[dkey] or 0) end
    end
    local lat = relief.lat or 0
    if lat < 0 then s = s + LATENCY_LAMBDA * lat end
    return s
end

-- Safe set 구성: latency LCB가 -C 이상인 action만 safe
--   equivalent to: cost UB = -relief.lat + β_t·‖φ‖_{V^{-1}} ≤ C
local function safe_set(ctx)
    local safe = {}
    for action, r in pairs(ctx.coef.relief) do
        local lat_mean = r.lat or 0
        local lat_ub   = (ctx.coef.cost_ub or {})[action] or 0   -- β_t·‖φ‖_{V^{-1}}
        local lat_lcb  = lat_mean - lat_ub
        if lat_lcb >= -LATENCY_SAFETY_C then
            safe[action] = true
        end
    end
    return safe
end

function decide(ctx)
    local p = ctx.coef.pressure
    local c = ctx.coef

    -- 1. Trigger check (policy_default와 동일)
    local trig = c.trigger
    if not (trig.tbt_degraded or trig.mem_low or trig.temp_high) then
        return {}
    end

    -- 2. Scalarization weight 계산 (docs/44)
    local w = domain_weights(p, ctx.history)
    local has_emergency = false
    for _, wv in pairs(w) do
        if wv >= EMERGENCY_WEIGHT then has_emergency = true; break end
    end

    local any_active = false
    for _, wv in pairs(w) do if wv > 0 then any_active = true; break end end
    if not any_active then return {} end

    -- 3. Safe set 구성
    local safe = safe_set(ctx)

    -- 4. Safe action 내에서 UCB argmax
    local best_action, best_ucb = nil, -math.huge
    for action, _ in pairs(safe) do
        local relief = c.relief[action]
        if relief then
            local reward_mean = compute_scalarized_reward(relief, w)
            -- UCB bonus: β_t·‖φ‖_{V^{-1}}
            -- 모든 dim에 동일 값이므로 active dim 하나의 ub만 쓰면 됨 (또는 weighted sum)
            local reward_ub = 0
            for dkey, wv in pairs(w) do
                if wv > 0 then
                    reward_ub = reward_ub + wv * ((c.relief_ub[action] or {})[dkey] or 0)
                end
            end
            -- latency penalty 항 uncertainty도 반영
            if (relief.lat or 0) < 0 then
                reward_ub = reward_ub + LATENCY_LAMBDA * ((c.relief_ub[action] or {}).lat or 0)
            end

            local ucb = reward_mean + reward_ub
            if ucb > best_ucb then
                best_action, best_ucb = action, ucb
            end
        end
    end

    -- 5. Seed safe action fallback: safe set이 비었거나 선택 실패
    if not best_action then
        -- Amani §3의 seed safe action = no-op (아무것도 바꾸지 않으면 latency 악화 없음)
        -- 로그로 관측 — 이 이벤트가 빈번하면 β_t가 너무 크거나 C가 너무 엄격
        sys.log("safe_lucb: empty safe set or no viable action, emitting no-op")
        return {}
    end

    -- 6. Action에 대응하는 EngineCommand 구성 (policy_default의 build_cmd 재사용)
    local lvl = detect_level_from_pressure(p)  -- docs/44 helper 재사용 가정
    local cmd = build_cmd(best_action, lvl, p.cpu or 0)
    if cmd then return { cmd } else return {} end
end
```

### 4.2 의사코드의 구현 주의사항

1. **`build_cmd` 재사용**: `policy_default.lua` v2.x의 `build_cmd(action_name, lvl, cpu_p)` helper를 공통 lib로 추출할 것 (`manager/scripts/lib/action_build.lua`).
2. **`detect_level_from_pressure`**: docs/44 §3.1에서 임계값 비교로 Level 결정 (Emergency > Critical > Warning > Normal).
3. **No-op 경로의 관측**: 매 tick `return {}`이 발생하면 `sys.log`를 통해 manager가 카운트할 수 있도록 (`empty_safe_set_count` 메트릭).
4. **UCB bonus의 dim 합산 방식**: 현재 구조는 모든 dim에 동일 scalar가 들어가므로 $\sum_d w_d \cdot \text{ub} = \text{ub} \cdot \sum_d w_d$로 단순화 가능. 향후 dim별 uncertainty 분리 시 generic 형태 유지.

### 4.3 Config (AdaptationConfig 확장)

```toml
# manager/config.toml 추가 필드

[adaptation.safe_bandit]
# Safe-LUCB 활성화 여부 (false면 coef.relief_ub/cost_ub/beta_t 주입 생략)
enabled = false

# Confidence radius 파라미터 (§2.4)
noise_std_r       = 0.1       # R: relief 관측 노이즈 std 추정
failure_prob_delta = 0.05     # δ: 1 - confidence level
feature_norm_l    = 3.6       # L: ‖φ‖_2 upper bound (√13 이므로 safety margin)
theta_norm_b      = 2.0       # B_μ: ‖θ*‖_2 upper bound
lambda_ridge      = 0.01      # λ: ridge regularization (P 초기값 100 → 1/100 = 0.01)
alpha_scale       = 1.0       # practical scaling factor (§2.4 empirical)
kappa_delay       = 0.1       # Pike-Burke delay penalty coef

# Safety constraint (§2.3)
latency_safety_c  = 0.30
```

---

## 5. $\beta_t$ 튜닝 전략

### 5.1 이론값 (baseline)

$\beta_0$: §2.4 수식을 그대로 계산하여 초기 guess로 사용. 예시 ($t = 100$, $D = 13$, $R = 0.1$, $\delta = 0.05$, $L = 3.6$, $B_\mu = 2.0$, $\lambda = 0.01$):

$$
\beta_0 \approx 0.1 \cdot \sqrt{2 \log 20 + 13 \log(1 + 1296000)} + 2 \cdot 0.1 \approx 0.1 \cdot 13.7 + 0.2 \approx 1.57
$$

### 5.2 Empirical scaling

실제 noise variance와 misspecification 때문에 이론값이 너무 크거나 작을 수 있다. 시뮬레이터 `manager/src/sim_run.rs`로 다음 지표를 측정하며 $\alpha$를 튜닝:

| 지표 | 의미 | 목표 |
|------|------|------|
| `empty_safe_set_rate` | safe set이 빈 tick 비율 | $< 5\%$ |
| `safety_violation_rate` | 실제 latency cost가 $C$ 초과한 tick 비율 | $< \delta = 5\%$ |
| `exploration_diversity` | action 선택 entropy | mean-only argmax 대비 $\ge 1.3\times$ |
| `scalarized_reward_total` | 누적 scalarized reward | mean-only argmax 대비 유지 또는 개선 |

**튜닝 순서**:

1. $\alpha = 1.0$으로 시작 → `safety_violation_rate` 측정
2. `safety_violation_rate > \delta`이면 $\alpha$ 증가 (더 pessimistic)
3. `empty_safe_set_rate > 10\%`이면 $\alpha$ 감소 (safe set이 너무 작음)
4. 최종: Pareto frontier (safety vs exploration) 위에서 선택

### 5.3 $\kappa_{\text{delay}}$ 튜닝

- in-flight queue 길이 분포 측정 (정상 상태에서 0~3, 버스트 시 up to 32)
- $\kappa = 0.1$이면 queue=10일 때 $\beta_t$가 $2\times$ 증가 — 이 정도가 합리적인지 시뮬레이터로 확인

---

## 6. 단계별 구현 계획

### 6.1 Phase 0: Indicator-weighted Scalarization (docs/44)

- **기간**: 1주
- **목표**: `policy_default.lua` v2.0.0으로 scalarization 도입
- **변경 파일**:
  - `manager/scripts/policy_default.lua` (로직 교체)
  - `docs/44_multi_domain_policy_design.md` (본 문서 현재 §7과 연동)
- **검증**: 시뮬레이터 trace (memory Emergency + compute Warning 동시) 에서 cpu 도메인 액션이 선택되는지
- **독립적 가치**: Phase 1~3 중단하더라도 Phase 0은 유지

### 6.2 Phase 1: Confidence Radius Infrastructure

- **기간**: 1주
- **목표**: `predict_with_ub()` + Lua `ctx.coef.relief_ub`/`cost_ub`/`beta_t`/`inflight_count` 노출
- **변경 파일**:
  - `manager/src/relief/mod.rs` (trait 확장)
  - `manager/src/relief/linear.rs` (`mahalanobis_norm`, `confidence_radius` 신규)
  - `manager/src/lua_policy.rs::build_ctx()` (Lua context 확장)
  - `manager/src/config.rs` (`AdaptationConfig::safe_bandit` 섹션)
  - **M1**: `EwmaReliefTable` → `OnlineLinearEstimator` 마이그레이션 (또는 dual-path 유지 선택)
- **검증**:
  - Unit test: `LinearModel::mahalanobis_norm`의 sanity (P identity일 때 $\|\phi\|_2$)
  - `confidence_radius`가 $t$ 증가에 따라 감소 (관측 많을수록 낮은 UB)
  - Lua에서 `ctx.coef.relief_ub[a][d]`가 0 이상
- **Spec 추가**: MGR-POL-301~305 (ID 할당 예정)
- **주의**: Phase 1만 완료하고 Phase 2 미착수 상태에서도 기존 `policy_default.lua`(docs/44)는 정상 동작해야 함 — 새 context 필드는 **optional**로 Lua에서 무시 가능해야 함.

### 6.3 Phase 2: Safe-LUCB Lua Policy + Simulator AB

- **기간**: 2주
- **목표**: `policy_safe_lucb.lua` 작성 + 시뮬레이터에서 docs/44와 비교
- **변경 파일**:
  - `manager/scripts/policy_safe_lucb.lua` (신규)
  - `manager/scripts/lib/action_build.lua` (공통 helper 추출)
  - `manager/src/sim_run.rs` (AB test 모드 또는 별도 시나리오)
- **검증**: §5.2의 4개 지표를 docs/44 대비 측정
- **Go/No-go 기준**:
  - `safety_violation_rate < \delta`
  - `scalarized_reward_total`이 docs/44 대비 $-5\%$ 이내
  - 둘 중 하나라도 불충족 시 Phase 3 중단

### 6.4 Phase 3: Production 승격 (조건부)

- **기간**: 진입 시 1주
- **목표**: `policy_default.lua`의 기본 정책을 Safe-LUCB로 교체 또는 feature flag 기반 전환
- **결정 근거**: Phase 2의 AB test 결과 + 최소 7일간 device baseline 검증
- **Rollback 경로**: `AdaptationConfig::safe_bandit.enabled = false`로 즉시 disable 가능 (Phase 1 infrastructure는 유지)

---

## 7. 참고 논문 및 불일치 사항

### 7.1 주 참고 논문

- **Amani, S., Alizadeh, M., & Thrampoulidis, C. (2019).** *Linear Stochastic Bandits Under Safety Constraints.* NeurIPS. — Safe-LUCB 원 논문. §4.1의 confidence radius, §4.2의 safe set 정의를 채택.
- **Russac, Y., Vernade, C., & Cappé, O. (2019).** *Weighted Linear Bandits for Non-Stationary Environments.* NeurIPS. — `forgetting_factor` 기반 non-stationary 확장. §2.4의 $\beta_t^{\text{NS}}$ 수식.
- **Pike-Burke, C., Agrawal, S., Szepesvari, C., & Grunewalder, S. (2018).** *Bandits with Delayed, Aggregated Anonymous Feedback.* ICML. — 3s observation delay에 따른 in-flight penalty. §2.7.
- **Abbasi-Yadkori, Y., Pál, D., & Szepesvári, C. (2011).** *Improved Algorithms for Linear Stochastic Bandits.* NeurIPS. — OFUL / LinUCB의 confidence ellipsoid. Amani가 확장한 기반.

### 7.2 논문 대비 불가피한 수정 (M1~M5)

`.agent/research/2026-04-15_amani2019_mapping.md`의 분석을 요약:

| ID | Amani 가정 | 우리 시스템 현실 | 대응 |
|----|----------|----------------|------|
| **M1** | 단일 global parameter $\theta^*$ | Per-action `LinearModel` | Per-action ridge matrix 유지 (이미 구조적으로 그러함). 단, `EwmaReliefTable` 경로 제거 필요. |
| **M2** | Safe set 내부 최적화가 LP/QP | Lua 스크립트에서 if 조건 분기 | Confidence radius를 Lua context에 노출하여 스크립트가 safe set 직접 구성 |
| **M3** | $c(a, \phi) = \phi^\top \mu^*$ single linear constraint | Latency가 relief vector의 한 차원 | 차원 분리: cost head = relief의 latency slot, pessimistic UB 별도 계산 |
| **M4** | Known seed safe action $a_0$ | 명시적 seed 없음 | No-op(빈 command)을 암묵적 seed로 사용. 정의상 latency 악화 없음. |
| **M5** | Instant feedback | 3s heartbeat delay, in-flight queue | Pike-Burke penalty로 $\beta_t$ 부풀리기 (§2.7) |

### 7.3 선택적 수정 (O1~O4)

Researcher 분석의 선택 사항 — 구현하지 않지만 향후 재검토:

- **O1**: Continuous action space로 확장 (예: throttle delay를 연속값으로) — 현재는 9-way discrete
- **O2**: Contextual Thompson sampling으로 UCB 대체 (Agrawal & Goyal 2013) — 경험적으로 UCB보다 낮은 regret
- **O3**: Meta-learning for cross-action feature sharing (GenAI: shared representation) — 데이터 효율성 향상
- **O4**: Conformal prediction으로 $\beta_t$ 대체 (empirical miscoverage 보장) — 이론 의존성 감소

### 7.4 명시적 불일치 (Assumption 위반)

| Amani 가정 | 우리 시스템 | 위반 정도 | 영향 |
|----------|------------|----------|------|
| A1 (선형성) | RLS linear model 사용 | 일치 | — |
| A2 (bounded noise, subgaussian) | Relief 관측이 subgaussian인지 불명 | 부분 위반 | $R$ 튜닝으로 완화 |
| A3 (bounded parameter $\|\theta^*\| \le B_\mu$) | 불명 | 부분 위반 | $B_\mu$ 튜닝 |
| A4 (bounded feature $\|\phi\| \le L$) | FEATURE_DIM=13, 값은 $[0, 1]$ 또는 정규화됨 | 대체로 만족 | $L = \sqrt{13}$ |
| A5 (ridge $\lambda > 0$) | 초기 P = $100 I$ | 만족 ($\lambda_{\text{eff}} = 0.01$) | — |
| A6 (continuous convex action) | Discrete $\|A\| = 9$ | **위반** | Per-arm ridge matrix로 우회 |
| A7 (stationarity) | Non-stationary | **위반** | Russac 2019로 확장 (§2.4) |
| A8 (known seed safe) | 없음 | 위반 | No-op을 암묵적 seed (§2.5) |

---

## 8. Regret Bound (참고)

Amani §5 Theorem 1 (보정):

$$
\mathrm{Regret}(T) \le O\!\left( D \sqrt{T} \log T \right) \quad \text{(stationary, continuous action)}
$$

우리 시스템 (non-stationary + discrete + delayed feedback) 에서는 이 bound가 **이론적으로 유지되지 않는다**. 그러나 다음 경험적 근거로 실용적 개선을 기대:

- **Mean-only baseline**은 regret bound 자체가 없음 (exploration 부재로 suboptimal lock-in 가능)
- **UCB bonus**는 log-factor 증가만 감수하고 exploration 보장
- **Safe set**은 constraint violation 확률을 $\delta$로 유한하게 bound (Amani Thm 2)

실측 regret은 시뮬레이터 AB (§6.3)로만 검증된다.

---

## 9. 미결 사항

### 9.1 차원 정합성 (6D vs 4D)

- `EwmaReliefTable` (6D: gpu/cpu/mem/therm/lat/qos) vs `OnlineLinearEstimator` (4D: compute/memory/thermal/latency)
- Phase 1 착수 시 재조사 — `types.rs::ReliefVector` 확장 vs 투영 매핑 결정

### 9.2 Inflight Penalty의 근거

- $\kappa = 0.1$은 heuristic
- Pike-Burke 원 논문의 delay-aware LinUCB 수식을 그대로 적용할지, 단순 scaling으로 대체할지

### 9.3 Multi-action 선택 시 UCB 합산 방식

- 현재 Safe-LUCB는 single action 선택 전제
- docs/44 / policy_default.lua는 여러 command를 동시 방출 가능
- Multi-arm combinatorial bandit (Chen et al. 2013)으로 확장 필요 여부 — Phase 2 시나리오에서 판단

### 9.4 Safe set 공집합 처리

- No-op을 seed로 사용하는 것이 적절한지
- 대안: 가장 보수적인 action (throttle) 강제 선택
- 시뮬레이터 관측 후 결정

### 9.5 Per-dim uncertainty vs scalar

- 현재 설계는 confidence bonus를 모든 dim에 동일 scalar로 노출
- 수학적으로는 `LinearModel.weights[d]` 별로 `V_{a,d}^{-1}`가 달라야 할 수 있음 — 그러나 현재 P matrix는 shared
- Shared P의 이론적 정당화: 모든 head가 동일한 feature로 regress하므로 Fisher information이 동일 — OK
- 단, bias는 별도 EMA이므로 추가 uncertainty가 존재 (무시 가능 수준)

---

## 10. 구현 작업 목록 (Implementer 위임용)

Phase 0 완료 후 (docs/44 §7 참조), 아래 작업을 Phase별로 위임:

### Phase 1 (Infrastructure)

1. `manager/src/relief/linear.rs`에 `LinearModel::mahalanobis_norm(&self, phi: &[f32]) -> f32` 추가
2. `manager/src/relief/linear.rs`에 `OnlineLinearEstimator::confidence_radius(&self, action: &ActionId, config: &ConfidenceConfig) -> f32` 추가
3. `manager/src/relief/mod.rs`의 `ReliefEstimator` trait에 `predict_with_ub()` default 구현 추가
4. `manager/src/config.rs`에 `ConfidenceConfig` + `AdaptationConfig::safe_bandit` 섹션 추가
5. `manager/src/lua_policy.rs::build_ctx()` 확장: `ctx.coef.relief_ub`, `ctx.coef.cost_ub`, `ctx.coef.beta_t`, `ctx.coef.inflight_count` 주입
6. **M1 마이그레이션**: `EwmaReliefTable` → `OnlineLinearEstimator` 교체 (또는 dual-path 결정)
7. Unit tests: mahalanobis norm identity, confidence radius monotonicity, Lua context 필드 존재
8. `spec/` 업데이트: MGR-POL-301~305 신규 ID 할당 (Architect)
9. `arch/` 업데이트: relief/ 컴포넌트 문서에 confidence radius 매핑 추가 (Architect)

### Phase 2 (Policy + AB Test)

1. `manager/scripts/lib/action_build.lua` 추출 (policy_default에서 공통화)
2. `manager/scripts/policy_safe_lucb.lua` 신규 작성 — §4.1 의사코드 기반
3. `manager/src/sim_run.rs`에 AB 비교 모드 추가 — 동일 trace로 두 정책 실행
4. 시뮬레이터 지표 수집: `empty_safe_set_rate`, `safety_violation_rate`, `exploration_diversity`, `scalarized_reward_total`
5. $\alpha$ 스윕 (e.g., $\{0.5, 1.0, 2.0, 4.0\}$) 후 Pareto optimal 선택
6. 결과 리포트: `.agent/research/2026-04-{XX}_safe_lucb_ab.md`

### Phase 3 (Production — 조건부)

1. Feature flag 기반 전환: `AdaptationConfig::safe_bandit.enabled = true`로 기본값 변경 여부 결정
2. Rollback 경로 문서화 (`docs/45` updates section)
3. 7일 baseline 수집 후 retention 확정

**구현 시 주의**:

- Phase 1 완료만으로는 Lua 정책이 변하지 않는다 (context 필드가 추가되지만 `policy_default.lua`는 기존 필드만 사용). 그러므로 Phase 1은 **순수 additive** 변경으로 유지할 것.
- Phase 2의 `policy_safe_lucb.lua`는 docs/44의 scalarization 로직을 **복사하지 말고** `lib/action_build.lua` 또는 유사 lib로 공통화한 후 require/reuse할 것 (dual source of truth 방지).
