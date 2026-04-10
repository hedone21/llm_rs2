-- policy_example.lua
-- Dynamic coefficient policy for llm_manager.
--
-- Uses ctx.coef (pressure, trigger, relief) provided by Rust.
-- Relief values are learned via EWMA at runtime.
--
-- Usage:
--   llm_manager --policy-script manager/scripts/policy_example.lua \
--               --transport unix:/tmp/llm.sock

function decide(ctx)
    local c = ctx.coef
    local t = c.trigger

    -- No contention evidence → no intervention
    if not t.tbt_degraded and not t.mem_low and not t.temp_high then
        -- Release active actions if pressure is low
        if #ctx.active > 0 then
            local p = c.pressure
            if p.gpu < 0.3 and p.memory < 0.3 and p.thermal < 0.3 then
                return {{type = "restore_defaults"}}
            end
        end
        return {}
    end

    -- Find the domain with highest pressure
    local p = c.pressure
    local domains = {gpu = p.gpu, cpu = p.cpu, memory = p.memory, thermal = p.thermal}
    local max_domain, max_val = nil, 0
    for k, v in pairs(domains) do
        if v > max_val then
            max_domain = k
            max_val = v
        end
    end

    if max_domain == nil then
        return {}
    end

    -- Map relief table keys to domain keys
    local domain_key = max_domain
    if domain_key == "memory" then domain_key = "mem" end
    if domain_key == "thermal" then domain_key = "therm" end

    -- Select the action with highest relief for the bottleneck domain,
    -- while respecting latency budget (lat >= -0.15)
    local best_action = nil
    local best_relief = -999

    for action, r in pairs(c.relief) do
        local relief_val = r[domain_key] or 0
        if relief_val > best_relief and (r.lat or 0) >= -0.15 then
            best_action = action
            best_relief = relief_val
        end
    end

    if best_action and best_relief > 0 then
        -- Build action with sensible defaults
        local cmd = {type = best_action}
        if best_action == "kv_evict_h2o" or best_action == "kv_evict_sliding"
           or best_action == "kv_merge_d2o" then
            cmd.keep_ratio = 0.5
        elseif best_action == "throttle" then
            cmd.delay_ms = 50
        elseif best_action == "set_target_tbt" then
            cmd.target_ms = 150
        elseif best_action == "layer_skip" then
            cmd.skip_ratio = 0.25
        elseif best_action == "switch_hw" then
            cmd.device = "cpu"
        elseif best_action == "kv_quant_dynamic" then
            cmd.target_bits = 4
        elseif best_action == "set_partition_ratio" then
            cmd.ratio = 0.5
        end
        return {cmd}
    end

    return {}
end
