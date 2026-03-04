/**
 * Runner — Benchmark execution panel.
 */

const Runner = (() => {
    let _pollInterval = null;

    function init() {
        document.getElementById('btn-run-benchmark').addEventListener('click', _startBenchmark);
    }

    async function _startBenchmark() {
        const backend = document.getElementById('run-backend').value;
        const prefillType = document.getElementById('run-prefill').value;
        const numTokens = parseInt(document.getElementById('run-tokens').value, 10);

        const btn = document.getElementById('btn-run-benchmark');
        btn.disabled = true;
        btn.textContent = '⏳ Starting...';

        try {
            const resp = await fetch('/api/benchmark/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    backend,
                    prefill_type: prefillType,
                    num_tokens: numTokens,
                }),
            });

            const data = await resp.json();

            if (!resp.ok) {
                alert(`Error: ${data.error}`);
                btn.disabled = false;
                btn.textContent = '▶ Start Benchmark';
                return;
            }

            // Show status panel, start polling
            document.getElementById('runner-status-panel').style.display = 'block';
            _startPolling();

        } catch (err) {
            alert(`Failed: ${err.message}`);
            btn.disabled = false;
            btn.textContent = '▶ Start Benchmark';
        }
    }

    function _startPolling() {
        if (_pollInterval) clearInterval(_pollInterval);
        _pollInterval = setInterval(_pollStatus, 1000);
        _pollStatus(); // immediate first poll
    }

    async function _pollStatus() {
        try {
            const resp = await fetch('/api/benchmark/status');
            const data = await resp.json();

            const dot = document.querySelector('#runner-indicator .status-dot');
            const text = document.getElementById('runner-status-text');
            const log = document.getElementById('runner-log');

            // Update dot
            dot.className = `status-dot ${data.status}`;
            text.textContent = `${data.status.toUpperCase()}`;
            if (data.params && data.params.backend) {
                text.textContent += ` — ${data.params.backend} / ${data.params.prefill_type} / ${data.params.num_tokens}tok`;
            }

            // Update log (show last 200 lines)
            const logLines = data.log || [];
            log.textContent = logLines.slice(-200).join('\n');
            log.scrollTop = log.scrollHeight;

            // Stop polling if done
            if (!data.running && data.status !== 'idle' && data.status !== 'starting') {
                clearInterval(_pollInterval);
                _pollInterval = null;

                const btn = document.getElementById('btn-run-benchmark');
                btn.disabled = false;
                btn.textContent = '▶ Start Benchmark';

                if (data.status === 'completed') {
                    // Reload profiles
                    setTimeout(() => App.reload(), 1000);
                }
            }
        } catch (err) {
            console.error('Poll error:', err);
        }
    }

    return { init };
})();
