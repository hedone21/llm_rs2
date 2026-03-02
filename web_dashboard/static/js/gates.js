const Gates = (() => {
    async function load() {
        try {
            const resp = await fetch('/api/gates');
            if (!resp.ok) {
                _renderEmpty();
                return;
            }
            const data = await resp.json();
            _renderCards(data.summary, data.overall_gate);
            _renderTable(data.tiers);
            _renderHistory(data.history || []);
        } catch (e) {
            console.error('Gates load error:', e);
            _renderEmpty();
        }
    }

    function _renderEmpty() {
        const cards = document.getElementById('gates-cards');
        if (cards) cards.innerHTML = '<p style="color:var(--text-dim);padding:20px;">No gate data available. Run <code>python scripts/update_test_status.py</code> first.</p>';
    }

    function _renderCards(summary, gate) {
        const cards = document.getElementById('gates-cards');
        if (!cards) return;
        const gateClass = gate === 'PASS' ? 'badge-pass' : 'badge-fail';
        cards.innerHTML = `
            <div class="summary-card">
                <div class="summary-label">Components</div>
                <div class="summary-value">${summary.total_components}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Pass Rate</div>
                <div class="summary-value">${summary.pass_rate.toFixed(1)}%</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Failed</div>
                <div class="summary-value">${summary.fail}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Overall Gate</div>
                <div class="summary-value"><span class="badge ${gateClass}">${gate}</span></div>
            </div>
        `;
    }

    function _renderTable(tiers) {
        const tbody = document.getElementById('gates-table-body');
        if (!tbody) return;

        const tierNames = { T1: 'Foundation', T2: 'Algorithm', T3: 'Backend', T4: 'Integration' };
        let html = '';

        for (const [tier, info] of Object.entries(tiers)) {
            html += `<tr class="tier-header"><td colspan="7">${tier} - ${info.name}</td></tr>`;
            for (const c of info.components) {
                const badgeClass = `badge-${c.gate.toLowerCase()}`;
                html += `<tr>
                    <td>${c.name}</td>
                    <td>${tier}</td>
                    <td>${c.maturity}</td>
                    <td>${c.total_tests}</td>
                    <td>${c.passed}</td>
                    <td>${c.failed}</td>
                    <td><span class="badge ${badgeClass}">${c.gate}</span></td>
                </tr>`;
            }
        }
        tbody.innerHTML = html;
    }

    function _renderHistory(history) {
        const el = document.getElementById('gates-history-chart');
        if (!el || !history.length) {
            if (el) el.innerHTML = '<p style="color:var(--text-dim);padding:20px;">No history data yet.</p>';
            return;
        }

        if (typeof Plotly === 'undefined') {
            el.innerHTML = '<p style="color:var(--text-dim);padding:20px;">Plotly not loaded.</p>';
            return;
        }

        const trace = {
            x: history.map(h => h.date),
            y: history.map(h => h.pass_rate),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Pass Rate',
            line: { color: '#4f6cf7', width: 2 },
            marker: { size: 6 }
        };

        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#c8cad0' },
            xaxis: { title: 'Date', gridcolor: 'rgba(200,202,208,0.1)' },
            yaxis: { title: 'Pass Rate (%)', range: [0, 105], gridcolor: 'rgba(200,202,208,0.1)' },
            margin: { t: 20, r: 20, b: 50, l: 50 },
            height: 300
        };

        Plotly.newPlot(el, [trace], layout, { responsive: true, displayModeBar: false });
    }

    return { load };
})();
