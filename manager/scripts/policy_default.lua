-- policy_default.lua
-- Production default policy for llm_manager.
--
-- Decision flow:
--   1. trigger 없으면 restore or no-op
--   2. Z_k 계산 (multi-threshold excess virtual queue)
--   3. Candidate 목록 구성 (single + joint)
--   4. DPP score argmax: Σ_k Z_k·r_k(a) − V·ℓ(a), hard floor 적용
--   5. Safety Override: Emergency 도메인 시 throttle 강제 추가
--
-- POLICY_META.version을 변경할 때마다 changelog 주석도 갱신
--
-- Changelog:
--   2.0.0 (2026-04-15): DPP 기반 재설계 (docs/46_dpp_policy_design.md)
--     - max-domain argmax → Σ_k Z_k·r_k − V·ℓ linear scalarization
--     - Z_k = multi-threshold excess virtual queue (raw pressure 제거)
--     - Latency: binary gate → hard floor + V·ℓ soft penalty 하이브리드
--     - Joint action registry (kv_evict_plus_quant, throttle_plus_layer_skip)
--     - Emergency throttle → DPP-external Safety Override layer 분리
--   1.0.1 (2026-04-15): pressure_level 임계값 정렬
--   1.0.0 (2026-04-15): initial production policy

POLICY_META = { name = "llm_default", version = "2.0.0" }

-- ── DPP 상수 (docs/46 §4) ──────────────────────────────────────────────────
local DPP = {
    V           = 1.0,   -- latency penalty weight
    C           = 0.30,  -- latency hard floor (normal)
    C_EMERGENCY = 0.50,  -- latency hard floor (emergency)
    W_WARN      = 1.0,   -- threshold excess weights (§4.2.3)
    W_CRIT      = 2.0,
    W_EMERG     = 4.0,
}

-- Per-domain threshold (Rust Monitor defaults에 정렬, §4.2.4)
-- mem: MemoryMonitorConfig — available 40/20/10% → used 60/80/90%
-- cpu/gpu: ComputeMonitorConfig warn=70%, crit=90%; Emergency는 정책 레이어 자체 정의
-- therm: ThermalMonitorConfig 60/75/85°C 정규화 (85°C = 1.0 기준)
local THETA = {
    mem   = { warn = 0.60, crit = 0.80, emerg = 0.90 },
    cpu   = { warn = 0.70, crit = 0.85, emerg = 0.95 },
    gpu   = { warn = 0.70, crit = 0.85, emerg = 0.95 },
    therm = { warn = 0.70, crit = 0.85, emerg = 0.95 },
}

-- DPP 도메인 키 → ctx.coef.pressure 키 매핑
local PRESSURE_KEY = { mem = "memory", cpu = "cpu", gpu = "gpu", therm = "thermal" }

-- Joint action registry (§4.6) — arity ≤ 2, 수동 등록만
-- NOTE: kv_evict_sliding + kv_quant_dynamic은 같은 kv_quality 배타 그룹에 속하므로
--       동시 발행 금지 (정확도 이중 훼손). 해당 조합은 등록하지 않는다.
local JOINT_ACTIONS = {
    throttle_plus_layer_skip = { components = { "throttle", "layer_skip" } },
}

-- pressure.memory 값에 따른 파라미터 테이블
-- p.memory = 1.0 - (available / total) 즉 used 비율 (0.0~1.0)
-- Rust MemoryMonitorConfig::default() 임계값과 정렬:
--   Emergency : available <= 10%  → used >= 90%  → p.memory >= 0.90
--   Critical  : available <= 20%  → used >= 80%  → p.memory >= 0.80
--   Warning   : available <= 40%  → used >= 60%  → p.memory >= 0.60
local function pressure_level(mem_pressure)
    if mem_pressure >= 0.90 then return "emergency"
    elseif mem_pressure >= 0.80 then return "critical"
    elseif mem_pressure >= 0.60 then return "warning"
    else return "normal"
    end
end

-- pressure level → keep_ratio / target_bits
local LEVEL_PARAMS = {
    emergency = { keep_ratio = 0.25, target_bits = 2  },
    critical  = { keep_ratio = 0.50, target_bits = 4  },
    warning   = { keep_ratio = 0.70, target_bits = 8  },
    normal    = { keep_ratio = 0.85, target_bits = 16 },
}

-- ctx.active에 name이 포함되어 있는지 확인
local function is_active(name, active)
    for _, a in ipairs(active) do
        if a == name then return true end
    end
    return false
end

-- action 이름과 pressure level로 command를 빌드
local function build_cmd(action, level, cpu_pressure)
    local p = LEVEL_PARAMS[level] or LEVEL_PARAMS.normal
    local cmd = { type = action }

    if action == "kv_evict_h2o" or action == "kv_evict_sliding"
       or action == "kv_merge_d2o" then
        cmd.keep_ratio = p.keep_ratio
    elseif action == "kv_quant_dynamic" then
        cmd.target_bits = p.target_bits
    elseif action == "throttle" then
        -- cpu_pressure에 비례, 최소 20ms, 20ms 단위 양자화
        -- (미세한 pressure 변동이 매 tick 다른 directive를 생성하지 않도록)
        local raw = math.floor(cpu_pressure * 200)
        cmd.delay_ms = math.max(math.floor(raw / 20) * 20, 20)
    elseif action == "set_target_tbt" then
        cmd.target_ms = 150
    elseif action == "layer_skip" then
        cmd.skip_ratio = 0.25
    elseif action == "switch_hw" then
        cmd.device = "cpu"
    elseif action == "set_partition_ratio" then
        cmd.ratio = 0.5
    end

    return cmd
end

-- Z_k: multi-threshold excess virtual queue (§4.2)
local function compute_Zk(pv, th)
    local ew = math.max(0, pv - th.warn)
    local ec = math.max(0, pv - th.crit)
    local ee = math.max(0, pv - th.emerg)
    return DPP.W_WARN * ew + DPP.W_CRIT * ec + DPP.W_EMERG * ee
end

-- Joint action relief 계산 (cold-start: component 선형 합 fallback, §4.6.6)
local function joint_relief(c, jkey)
    if c.relief[jkey] ~= nil then return c.relief[jkey] end
    local spec = JOINT_ACTIONS[jkey]
    if not spec then return nil end
    local sum = { memory = 0, cpu = 0, gpu = 0, thermal = 0, lat = 0 }
    for _, comp in ipairs(spec.components) do
        local r = c.relief[comp] or {}
        for k in pairs(sum) do sum[k] = sum[k] + (r[k] or 0) end
    end
    return sum
end

-- Joint action active 검사: component 중 하나라도 active면 true
local function is_active_any(name, active)
    local spec = JOINT_ACTIONS[name]
    if spec then
        for _, comp in ipairs(spec.components) do
            if is_active(comp, active) then return true end
        end
        return false
    end
    return is_active(name, active)
end

function decide(ctx)
    -- Joint action 검증 (ctx.is_joint_valid가 있을 때만 실행)
    if ctx.is_joint_valid then
        for jkey, jspec in pairs(JOINT_ACTIONS) do
            if not ctx.is_joint_valid(jspec.components) then
                error("Invalid joint action '" .. jkey .. "': conflicting components")
            end
        end
    end

    local c = ctx.coef
    local t = c.trigger
    local p = c.pressure

    -- 1. trigger 없으면 restore or no-op
    --    active가 있으면 무조건 restore (임의 0.3 임계값 제거)
    if not t.tbt_degraded and not t.mem_low and not t.temp_high then
        if #ctx.active > 0 then
            return {{ type = "restore_defaults" }}
        end
        return {}
    end

    -- 2. Z_k 계산 (multi-threshold excess virtual queue, docs/46 §4.2)
    local Z = {}
    -- has_compute_emergency: Safety Override predicate (cpu/gpu/therm만, §4.7.2)
    --   throttle은 compute 부하를 낮추는 액션이므로 메모리 emergency에는 사용 안 함
    -- any_emergency: latency hard floor 완화 predicate (모든 도메인)
    --   어느 도메인이든 emergency면 latency를 더 양보하여 강한 조치를 허용
    local has_compute_emergency = false
    local any_emergency = false
    for dkey, pkey in pairs(PRESSURE_KEY) do
        local pv = p[pkey] or 0
        Z[dkey] = compute_Zk(pv, THETA[dkey])
        if pv >= THETA[dkey].emerg then
            any_emergency = true
            if dkey ~= "mem" then has_compute_emergency = true end
        end
    end

    -- 모든 Z_k = 0 → 개입 불필요
    local any_z = false
    for _, zv in pairs(Z) do if zv > 0 then any_z = true; break end end
    if not any_z then
        if #ctx.active > 0 then return {{ type = "restore_defaults" }} end
        return {}
    end

    -- 3. pressure level 결정 (build_cmd 파라미터용; 기존 함수 재사용)
    local mem_p = p.memory or 0
    local lvl   = pressure_level(mem_p)
    local cpu_p = p.cpu or 0

    -- 4. Safe set: latency hard floor (Emergency 시 완화, §4.4)
    local lat_floor = any_emergency and -DPP.C_EMERGENCY or -DPP.C

    -- 5. Candidate 목록 (single action + joint action, §4.6)
    local candidates = {}
    for action, r in pairs(c.relief) do
        -- joint registry에 속한 single key는 제외 (joint이 이미 포함)
        if JOINT_ACTIONS[action] == nil then
            table.insert(candidates, { name = action, relief = r })
        end
    end
    for jkey in pairs(JOINT_ACTIONS) do
        local jr = joint_relief(c, jkey)
        if jr ~= nil then
            table.insert(candidates, { name = jkey, relief = jr, is_joint = true })
        end
    end

    -- 6. DPP score argmax: score = Σ_k Z_k·r_k(a) − V·max(-lat,0) (§4.1)
    local best_action, best_score = nil, -math.huge
    for _, cand in ipairs(candidates) do
        local r   = cand.relief
        local lat = r.lat or 0
        -- hard floor 검사 (safe set, 원칙 3)
        if lat >= lat_floor then
            -- resource term: Σ_k Z_k · r[domain_k]
            local resource_term = 0
            for dkey, zv in pairs(Z) do
                if zv > 0 then
                    resource_term = resource_term + zv * (r[PRESSURE_KEY[dkey]] or 0)
                end
            end
            -- latency soft penalty: V · max(-lat, 0)
            local latency_penalty = (lat < 0) and (DPP.V * (-lat)) or 0
            local score = resource_term - latency_penalty

            if score > best_score then
                best_action = cand.name
                best_score  = score
            end
        end
    end

    -- 7. Primary commands (DPP output)
    local primary_cmds = {}
    if best_action ~= nil and best_score > 0
       and not is_active_any(best_action, ctx.active) then
        if JOINT_ACTIONS[best_action] ~= nil then
            -- Joint action: 각 component를 별도 cmd로 빌드
            for _, comp in ipairs(JOINT_ACTIONS[best_action].components) do
                table.insert(primary_cmds, build_cmd(comp, lvl, cpu_p))
            end
        else
            table.insert(primary_cmds, build_cmd(best_action, lvl, cpu_p))
        end
    end

    -- 8. Safety Override (DPP 외부 레이어, §4.7)
    -- Compute/thermal emergency 시 DPP 결과와 무관하게 throttle 강제 추가.
    -- 메모리 emergency는 제외 — throttle은 compute 부하를 낮추는 액션이며
    -- 메모리는 kv_evict 계열로 처리해야 함.
    -- NOTE: commands > 1 → lua_policy.rs가 observation을 clear함 (observation skip 자동 처리)
    if has_compute_emergency and not is_active("throttle", ctx.active) then
        local has_thr = false
        for _, cmd in ipairs(primary_cmds) do
            if cmd.type == "throttle" then has_thr = true; break end
        end
        if not has_thr then
            table.insert(primary_cmds, build_cmd("throttle", lvl, cpu_p))
        end
    end

    return primary_cmds
end
