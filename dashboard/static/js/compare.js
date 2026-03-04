/**
 * Compare — Multi-profile overlay comparison.
 */

const Compare = (() => {

    /**
     * Load and render comparison for the given profile IDs.
     */
    async function loadAndRender(ids) {
        const headerEl = document.getElementById('compare-header');
        const summaryEl = document.getElementById('compare-summary');
        const chartsEl = document.getElementById('compare-charts');

        headerEl.innerHTML = '<h2>⏳ Loading comparison...</h2>';
        summaryEl.innerHTML = '';
        chartsEl.innerHTML = '';

        try {
            const resp = await fetch(`/api/compare?ids=${ids.join(',')}`);
            if (!resp.ok) throw new Error(`API error: ${resp.status}`);
            const data = await resp.json();

            headerEl.innerHTML = `<h2>Comparing ${data.profiles.length} Profiles</h2>`;
            _renderSummaryTable(summaryEl, data.profiles);
            _renderOverlayCharts(chartsEl, data.profiles);
        } catch (err) {
            headerEl.innerHTML = `<p class="placeholder-text">Error: ${err.message}</p>`;
        }
    }

    function _renderSummaryTable(container, profiles) {
        let html = `<table class="compare-table"><thead><tr>
            <th>Profile</th><th>Backend</th><th>Device</th><th>Prefill</th><th>Tokens</th>
            <th>TTFT (ms)</th><th>TBT (ms)</th><th>T/s</th><th>Start Temp</th><th>Max Temp</th>
        </tr></thead><tbody>`;

        for (const p of profiles) {
            const r = p.results || {};
            const m = p.metadata || {};
            const t = p.thermal || {};
            html += `<tr>
                <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;" title="${p.id}">${p.id.substring(0, 30)}...</td>
                <td><span class="badge badge-${m.backend || ''}">${m.backend || '—'}</span></td>
                <td><span class="badge" style="background:transparent;color:var(--text);border:1px solid var(--border);">${m.device || 'Unknown'}</span></td>
                <td>${m.prefill_type || '—'}</td>
                <td>${m.num_tokens ?? '—'}</td>
                <td>${r.ttft_ms != null ? r.ttft_ms.toFixed(1) : 'N/A'}</td>
                <td>${r.tbt_ms != null ? r.tbt_ms.toFixed(2) : 'N/A'}</td>
                <td>${r.tokens_per_sec != null ? r.tokens_per_sec.toFixed(1) : 'N/A'}</td>
                <td>${t.start_temp ?? '—'}</td>
                <td>${t.max_temp ?? '—'}</td>
            </tr>`;
        }

        html += '</tbody></table>';
        container.innerHTML = html;
    }

    function _renderOverlayCharts(container, profiles) {
        // Determine common timeseries fields
        const allFields = new Set();
        for (const p of profiles) {
            for (const fd of (p.field_descriptors || [])) {
                if (fd.chart !== 'multi_line') { // Skip multi_line for overlay simplicity
                    allFields.add(fd.key);
                }
            }
        }

        const fieldColors = [
            '#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6',
            '#06b6d4', '#ec4899', '#a3e635',
        ];

        for (const fieldKey of allFields) {
            const div = document.createElement('div');
            div.className = 'card';
            div.style.marginBottom = '16px';
            container.appendChild(div);

            const chartDiv = document.createElement('div');
            div.appendChild(chartDiv);

            const traces = [];
            profiles.forEach((p, idx) => {
                const ts = p.timeseries || [];
                if (ts.length === 0) return;

                // Use relative time (seconds from first sample)
                const t0 = new Date(ts[0].timestamp).getTime();
                const relTimes = ts.map(s => (new Date(s.timestamp).getTime() - t0) / 1000);

                const values = ts.map(s => s[fieldKey]);
                if (values.every(v => v == null)) return;

                const label = `${p.metadata?.backend || '?'}_${p.metadata?.prefill_type || '?'}_${p.metadata?.num_tokens || '?'}tok`;

                traces.push({
                    x: relTimes,
                    y: values,
                    name: label,
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: fieldColors[idx % fieldColors.length], width: 2 },
                });
            });

            if (traces.length === 0) {
                div.remove();
                continue;
            }

            // Find the field descriptor for the axis label
            const desc = profiles[0]?.field_descriptors?.find(fd => fd.key === fieldKey) || {};

            const layout = {
                ...Presenter.PLOTLY_LAYOUT_DEFAULTS,
                height: 280,
                title: { text: desc.label || fieldKey, font: { size: 14 } },
                xaxis: { ...Presenter.PLOTLY_LAYOUT_DEFAULTS.xaxis, title: 'Time (s)' },
                yaxis: {
                    ...Presenter.PLOTLY_LAYOUT_DEFAULTS.yaxis,
                    title: desc.axis_label || `${desc.label || fieldKey} (${desc.unit || ''})`,
                },
            };

            Plotly.newPlot(chartDiv, traces, layout, { responsive: true });
        }
    }

    return { loadAndRender };
})();
