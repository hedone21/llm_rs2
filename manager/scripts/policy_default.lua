-- policy_default.lua
-- Production default policy for llm_manager.
--
-- Decision flow:
--   1. trigger 없으면 restore or no-op
--   2. max-pressure 도메인 선정
--   3. relief argmax (lat constraint >= -0.20)
--   4. 파라미터는 pressure level에 따라 동적화
--   5. Emergency(>=0.93)에서만 보조 throttle 추가 (observation 포기)
--
-- POLICY_META.version을 변경할 때마다 changelog 주석도 갱신
--
-- Changelog:
--   1.0.0 (2026-04-15): initial production policy
--     - 3단계 pressure level (Warning/Critical/Emergency)
--     - keep_ratio / target_bits / throttle delay_ms 동적화
--     - active 중복 가드로 action 순환 해소
--     - restore_defaults 조건 단순화 (0.3 임계값 제거)
--     - Emergency에서 보조 throttle 추가 (relief observation 포기 허용)

POLICY_META = { name = "llm_default", version = "1.0.0" }

-- pressure.memory 값에 따른 파라미터 테이블
-- level: "warning"(0.70~), "critical"(0.85~), "emergency"(0.93~)
local function pressure_level(mem_pressure)
    if mem_pressure >= 0.93 then return "emergency"
    elseif mem_pressure >= 0.85 then return "critical"
    elseif mem_pressure >= 0.70 then return "warning"
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
        -- cpu_pressure에 비례, 최소 20ms
        local raw = math.floor(cpu_pressure * 200)
        cmd.delay_ms = math.max(raw, 20)
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

function decide(ctx)
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

    -- 2. max-pressure 도메인 선정 (cpu 포함, tie-break 알파벳순)
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

    if max_domain == nil then
        return {}
    end

    -- relief 테이블 lookup 키 변환 (memory→mem, thermal→therm)
    local domain_key = max_domain
    if domain_key == "memory"  then domain_key = "mem"   end
    if domain_key == "thermal" then domain_key = "therm" end

    -- 3. pressure level 결정 (memory 기준)
    local mem_p  = p.memory or 0
    local lvl    = pressure_level(mem_p)
    local cpu_p  = p.cpu    or 0

    -- Normal 수준이면 개입 없음 (trigger false 분기와 별개로 정밀 차단)
    if lvl == "normal" then
        return {}
    end

    -- 4. relief argmax: lat constraint >= -0.20
    local best_action = nil
    local best_relief = -999

    for action, r in pairs(c.relief) do
        local relief_val = r[domain_key] or 0
        local better = relief_val > best_relief
        local tied   = (relief_val == best_relief) and (best_action ~= nil) and (action < best_action)
        if (better or tied) and (r.lat or 0) >= -0.20 then
            best_action = action
            best_relief = relief_val
        end
    end

    if not best_action or best_relief <= 0 then
        return {}
    end

    -- active 중복 가드: 이미 동일 action이 active면 스킵
    -- (KvQuantDynamic→KvEvictH2o→KvEvictSliding→KvMergeD2o 순환 방지)
    if is_active(best_action, ctx.active) then
        return {}
    end

    local primary_cmd = build_cmd(best_action, lvl, cpu_p)

    -- 5. Emergency에서만 보조 throttle 추가
    --    NOTE: commands가 2개이면 relief observation이 큐잉되지 않음.
    --    Emergency에서는 즉각적 압박 완화가 우선이므로 이를 허용한다.
    if lvl == "emergency" then
        -- throttle이 primary로 이미 선정된 경우엔 중복 추가하지 않음
        if best_action ~= "throttle" and not is_active("throttle", ctx.active) then
            local throttle_delay = math.max(math.floor(cpu_p * 200), 20)
            local aux_throttle = { type = "throttle", delay_ms = throttle_delay }
            return { primary_cmd, aux_throttle }
        end
    end

    return { primary_cmd }
end
