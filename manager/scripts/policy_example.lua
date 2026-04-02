-- policy_example.lua
-- Simple memory/thermal-based policy example for llm_manager.
--
-- This script is loaded once at startup. The Lua VM persists across calls,
-- so global variables (like ema_temp below) retain their values.
--
-- Usage:
--   llm_manager --policy-script manager/scripts/policy_example.lua \
--               --transport unix:/tmp/llm.sock

local ema_temp = nil

function decide(ctx)
    local actions = {}

    local temp = sys.thermal(0)
    local mem = sys.meminfo()

    -- EMA temperature (exponential moving average)
    if ema_temp == nil then
        ema_temp = temp
    else
        ema_temp = 0.2 * temp + 0.8 * ema_temp
    end

    -- Memory danger: available < 15% of total
    if mem.total > 0 and (mem.available / mem.total) < 0.15 then
        table.insert(actions, {type = "kv_evict_h2o", keep_ratio = 0.5})
        return actions
    end

    -- Thermal danger: EMA > 42 C
    if ema_temp > 42 then
        table.insert(actions, {type = "set_target_tbt", target_ms = 200})
        return actions
    end

    -- Normal: release active actions if conditions are good
    if #ctx.active > 0 and ema_temp < 38 then
        if mem.total > 0 and (mem.available / mem.total) > 0.3 then
            table.insert(actions, {type = "restore_defaults"})
            return actions
        end
    end

    -- No action needed
    return actions
end
