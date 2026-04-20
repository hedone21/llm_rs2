-- Neutral no-op policy for benchmarking: never emits commands.
function decide(ctx)
    return {}
end
