/**
 * Presenter — Schema-driven Plotly chart builder.
 *
 * Reads field descriptors from the API and dynamically creates
 * Plotly subplots.  Adding a new timeseries field on the backend
 * automatically produces a new subplot here.
 */

const Presenter = (() => {
    const PLOTLY_LAYOUT_DEFAULTS = {
        paper_bgcolor: '#1c1f2e',
        plot_bgcolor: '#161822',
        font: { family: 'Inter, sans-serif', color: '#8b8fa3', size: 11 },
        margin: { l: 60, r: 60, t: 30, b: 40 },
        hovermode: 'x unified',
        showlegend: true,
        legend: { orientation: 'h', y: -0.15, font: { size: 10 } },
        xaxis: {
            gridcolor: '#2a2d3e',
            zerolinecolor: '#2a2d3e',
        },
        yaxis: {
            gridcolor: '#2a2d3e',
            zerolinecolor: '#2a2d3e',
        },
    };

    const EVENT_COLORS = {
        Load: 'rgba(156,163,175,0.22)',
        Prefill: 'rgba(245,158,11,0.22)',
        Decode: 'rgba(34,197,94,0.22)',
    };

    const EVENT_LABEL_COLORS = {
        Load: '#9ca3af',
        Prefill: '#f59e0b',
        Decode: '#22c55e',
    };

    // CPU core color palette
    const CORE_COLORS = [
        '#f87171', '#fb923c', '#fbbf24', '#a3e635',
        '#34d399', '#22d3ee', '#818cf8', '#e879f9',
    ];

    /**
     * Group field descriptors by their `group` key.
     */
    function groupFields(descriptors) {
        const groups = {};
        for (const desc of descriptors) {
            const g = desc.group || 'other';
            if (!groups[g]) groups[g] = [];
            groups[g].push(desc);
        }
        return groups;
    }

    /**
     * Parse events into region spans (shapes) and labels (annotations)
     * using relative-second x coordinates.
     */
    function buildEventShapesAndAnnotations(events, t0ms) {
        const evMap = {};
        for (const e of events) {
            evMap[e.name] = (new Date(e.timestamp).getTime() - t0ms) / 1000;
        }

        const regions = [
            { start: 'ModelLoadStart', end: 'PrefillStart', label: 'Load', color: EVENT_COLORS.Load, labelColor: EVENT_LABEL_COLORS.Load },
            { start: 'PrefillStart', end: 'DecodingStart', label: 'Prefill', color: EVENT_COLORS.Prefill, labelColor: EVENT_LABEL_COLORS.Prefill },
            { start: 'DecodingStart', end: 'End', label: 'Decode', color: EVENT_COLORS.Decode, labelColor: EVENT_LABEL_COLORS.Decode },
        ];

        const shapes = [];
        const annotations = [];
        for (const r of regions) {
            if (evMap[r.start] != null && evMap[r.end] != null) {
                shapes.push({
                    type: 'rect',
                    xref: 'x', yref: 'paper',
                    x0: evMap[r.start], x1: evMap[r.end],
                    y0: 0, y1: 1,
                    fillcolor: r.color,
                    line: { width: 0 },
                });
                // Label annotation centered in the region, at top
                annotations.push({
                    x: (evMap[r.start] + evMap[r.end]) / 2,
                    y: 1.0,
                    xref: 'x', yref: 'paper',
                    text: `<b>${r.label}</b>`,
                    showarrow: false,
                    font: { size: 11, color: r.labelColor },
                    yanchor: 'bottom',
                });
            }
        }
        return { shapes, annotations };
    }

    /**
     * Render a full profile detail with subplots.
     */
    function renderDetailCharts(containerId, profileData) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';

        const { timeseries, field_descriptors, events = [] } = profileData;
        if (!timeseries || timeseries.length === 0) {
            container.innerHTML = '<p class="placeholder-text">No timeseries data.</p>';
            return;
        }

        // Convert to relative seconds for a clean x-axis
        const t0ms = new Date(timeseries[0].timestamp).getTime();
        const relSeconds = timeseries.map(s => (new Date(s.timestamp).getTime() - t0ms) / 1000);

        const groups = groupFields(field_descriptors);
        const { shapes, annotations } = buildEventShapesAndAnnotations(events, t0ms);

        // Determine subplot order
        const groupOrder = ['thermal', 'cpu_freq', 'gpu_freq', 'memory', 'cpu_load', 'gpu_load'];
        const sortedGroups = Object.keys(groups).sort((a, b) => {
            const ia = groupOrder.indexOf(a);
            const ib = groupOrder.indexOf(b);
            if (ia === -1 && ib === -1) return a.localeCompare(b);
            if (ia === -1) return 1;
            if (ib === -1) return -1;
            return ia - ib;
        });

        const numSubplots = sortedGroups.length;
        const traces = [];
        const layout = {
            ...PLOTLY_LAYOUT_DEFAULTS,
            height: numSubplots * 180 + 60,
            grid: { rows: numSubplots, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
            shapes: [],
            annotations: [],
        };

        // Calculate max time from both timeseries and events
        let maxRelTime = relSeconds[relSeconds.length - 1];
        events.forEach(e => {
            const t = (new Date(e.timestamp).getTime() - t0ms) / 1000;
            if (t > maxRelTime) maxRelTime = t;
        });

        // Add a dummy invisible trace to force the x-axis range to cover all events
        traces.push({
            x: [0, maxRelTime * 1.05], // 5% padding
            y: [0, 0],
            mode: 'markers',
            marker: { opacity: 0 },
            showlegend: false,
            hoverinfo: 'skip',
            xaxis: 'x', // Use the first x-axis (master)
            yaxis: 'y',
        });


        sortedGroups.forEach((groupName, idx) => {
            const fields = groups[groupName];
            const axisIdx = idx + 1;
            const xAxisKey = `x${axisIdx}`;
            const yAxisKey = `y${axisIdx}`;

            // Find axis label from first field with one
            const axisLabel = fields.find(f => f.axis_label)?.axis_label || groupName;

            // Plotly expects 'xaxis' for the first axis, and 'xaxis2', 'xaxis3', etc. for others
            const key = axisIdx === 1 ? 'xaxis' : `xaxis${axisIdx}`;

            layout[key] = {
                ...PLOTLY_LAYOUT_DEFAULTS.xaxis,
                type: 'linear', // Force linear axis to prevent categorical inference
                showticklabels: idx === numSubplots - 1,
                matches: idx > 0 ? 'x' : undefined,
                title: idx === numSubplots - 1 ? { text: 'Time (s)', font: { size: 11 } } : undefined,
                tickfont: { size: 10 },
            };
            layout[`yaxis${axisIdx}`] = {
                ...PLOTLY_LAYOUT_DEFAULTS.yaxis,
                title: { text: axisLabel, font: { size: 11 } },
            };

            // Add event shapes to each subplot
            for (const shape of shapes) {
                layout.shapes.push({
                    ...shape,
                    xref: xAxisKey,
                    yref: `${yAxisKey} domain`,
                });
            }
            // Add labels only to the first subplot
            if (idx === 0) {
                for (const ann of annotations) {
                    layout.annotations.push({
                        ...ann,
                        xref: xAxisKey,
                        yref: `${yAxisKey} domain`,
                    });
                }
            }

            for (const field of fields) {
                if (field.chart === 'multi_line') {
                    // E.g., cpu_freqs_khz → one trace per core
                    const firstSample = timeseries.find(s => s[field.key] != null);
                    if (!firstSample) continue;
                    const numCores = Array.isArray(firstSample[field.key]) ? firstSample[field.key].length : 0;

                    for (let c = 0; c < numCores; c++) {
                        traces.push({
                            x: relSeconds,
                            y: timeseries.map(s => {
                                const arr = s[field.key];
                                return arr && arr[c] != null ? arr[c] : null;
                            }),
                            name: `CPU${c}`,
                            type: 'scatter',
                            mode: 'lines',
                            line: { color: CORE_COLORS[c % CORE_COLORS.length], width: 1.5 },
                            xaxis: xAxisKey,
                            yaxis: yAxisKey,
                            legendgroup: `cpu_freq`,
                            showlegend: idx === sortedGroups.indexOf('cpu_freq'),
                        });
                    }
                } else if (field.chart === 'multi_line_dict') {
                    // E.g., temps -> { cpu: 70, gpu: 60 }
                    // Collect all unique dict keys across the timeseries
                    const allKeys = new Set();
                    timeseries.forEach(s => {
                        const obj = s[field.key];
                        if (obj) {
                            Object.keys(obj).forEach(k => allKeys.add(k));
                        }
                    });

                    let c = 0;
                    for (const k of Array.from(allKeys).sort()) {
                        traces.push({
                            x: relSeconds,
                            y: timeseries.map(s => {
                                const obj = s[field.key];
                                return obj && obj[k] != null ? obj[k] : null;
                            }),
                            name: k,
                            type: 'scatter',
                            mode: 'lines',
                            line: { color: CORE_COLORS[c % CORE_COLORS.length], width: 1.5 },
                            xaxis: xAxisKey,
                            yaxis: yAxisKey,
                            legendgroup: field.group,
                            showlegend: idx === sortedGroups.indexOf(field.group),
                        });
                        c++;
                    }
                } else {
                    // Standard line trace
                    traces.push({
                        x: relSeconds,
                        y: timeseries.map(s => s[field.key]),
                        name: field.label,
                        type: 'scatter',
                        mode: 'lines',
                        line: {
                            color: field.color || '#8b8fa3',
                            width: 2,
                            dash: field.style || 'solid',
                        },
                        xaxis: xAxisKey,
                        yaxis: yAxisKey,
                    });
                }
            }
        });

        Plotly.newPlot(container, traces, layout, { responsive: true, displayModeBar: true });
    }


    /**
     * Render overview charts on the Overview tab.
     */
    function renderOverviewCharts(profiles) {
        _renderBackendPerfChart(profiles);
        _renderTokenScalingChart(profiles);
        _renderThermalPerfChart(profiles);
    }

    function _renderBackendPerfChart(profiles) {
        const valid = profiles.filter(p => p.results.tokens_per_sec != null);
        const cpu = valid.filter(p => p.metadata.backend === 'cpu');
        const opencl = valid.filter(p => p.metadata.backend === 'opencl');

        const traces = [
            {
                y: cpu.map(p => p.results.tokens_per_sec),
                name: 'CPU',
                type: 'box',
                marker: { color: '#8b5cf6' },
                boxpoints: 'all',
                jitter: 0.3,
                pointpos: -1.5,
            },
            {
                y: opencl.map(p => p.results.tokens_per_sec),
                name: 'OpenCL',
                type: 'box',
                marker: { color: '#06b6d4' },
                boxpoints: 'all',
                jitter: 0.3,
                pointpos: -1.5,
            },
        ];

        const layout = {
            ...PLOTLY_LAYOUT_DEFAULTS,
            height: 300,
            yaxis: { ...PLOTLY_LAYOUT_DEFAULTS.yaxis, title: 'Tokens/sec' },
        };

        Plotly.newPlot('chart-backend-perf', traces, layout, { responsive: true });
    }

    function _renderTokenScalingChart(profiles) {
        const valid = profiles.filter(p => p.results.tbt_ms != null && p.metadata.num_tokens != null);
        const cpu = valid.filter(p => p.metadata.backend === 'cpu');
        const opencl = valid.filter(p => p.metadata.backend === 'opencl');

        const traces = [
            {
                x: cpu.map(p => p.metadata.num_tokens),
                y: cpu.map(p => p.results.tbt_ms),
                name: 'CPU',
                mode: 'markers',
                type: 'scatter',
                marker: { color: '#8b5cf6', size: 6 },
            },
            {
                x: opencl.map(p => p.metadata.num_tokens),
                y: opencl.map(p => p.results.tbt_ms),
                name: 'OpenCL',
                mode: 'markers',
                type: 'scatter',
                marker: { color: '#06b6d4', size: 6 },
            },
        ];

        const layout = {
            ...PLOTLY_LAYOUT_DEFAULTS,
            height: 300,
            xaxis: { ...PLOTLY_LAYOUT_DEFAULTS.xaxis, title: 'Token Count', type: 'log' },
            yaxis: { ...PLOTLY_LAYOUT_DEFAULTS.yaxis, title: 'Avg TBT (ms)' },
        };

        Plotly.newPlot('chart-token-scaling', traces, layout, { responsive: true });
    }

    function _renderThermalPerfChart(profiles) {
        const valid = profiles.filter(p =>
            p.results.tokens_per_sec != null && p.thermal.start_temp != null
        );

        const traces = [{
            x: valid.map(p => p.thermal.start_temp),
            y: valid.map(p => p.results.tokens_per_sec),
            text: valid.map(p => `${p.metadata.backend} | ${p.metadata.prefill_type} | ${p.metadata.num_tokens}tok`),
            mode: 'markers',
            type: 'scatter',
            marker: {
                color: valid.map(p => p.metadata.backend === 'cpu' ? '#8b5cf6' : '#06b6d4'),
                size: 8,
            },
        }];

        const layout = {
            ...PLOTLY_LAYOUT_DEFAULTS,
            height: 300,
            showlegend: false,
            xaxis: { ...PLOTLY_LAYOUT_DEFAULTS.xaxis, title: 'Start Temp (°C)' },
            yaxis: { ...PLOTLY_LAYOUT_DEFAULTS.yaxis, title: 'Tokens/sec' },
        };

        Plotly.newPlot('chart-thermal-perf', traces, layout, { responsive: true });
    }


    /**
     * Render trends chart.
     */
    function renderTrendsChart(profiles, metric, backendFilter) {
        let filtered = profiles.filter(p => p.results[metric] != null && p.metadata.date != null);
        if (backendFilter) {
            filtered = filtered.filter(p => p.metadata.backend === backendFilter);
        }

        // Sort by date
        filtered.sort((a, b) => (a.metadata.date || '').localeCompare(b.metadata.date || ''));

        // Group by config: backend + prefill + tokens
        const groups = {};
        for (const p of filtered) {
            const key = `${p.metadata.backend}_${p.metadata.prefill_type}_${p.metadata.num_tokens}tok`;
            if (!groups[key]) groups[key] = [];
            groups[key].push(p);
        }

        const metricLabels = {
            tokens_per_sec: 'Tokens/sec',
            tbt_ms: 'Avg TBT (ms)',
            ttft_ms: 'TTFT (ms)',
        };

        const traces = Object.entries(groups).map(([key, ps]) => ({
            x: ps.map(p => p.metadata.date),
            y: ps.map(p => p.results[metric]),
            name: key,
            type: 'scatter',
            mode: 'lines+markers',
            marker: { size: 5 },
        }));

        const layout = {
            ...PLOTLY_LAYOUT_DEFAULTS,
            height: 500,
            yaxis: { ...PLOTLY_LAYOUT_DEFAULTS.yaxis, title: metricLabels[metric] || metric },
            xaxis: { ...PLOTLY_LAYOUT_DEFAULTS.xaxis, title: 'Date' },
        };

        Plotly.newPlot('trends-chart', traces, layout, { responsive: true });
    }


    // Public API
    return {
        renderDetailCharts,
        renderOverviewCharts,
        renderTrendsChart,
        groupFields,
        PLOTLY_LAYOUT_DEFAULTS,
        CORE_COLORS,
        EVENT_COLORS,
        buildEventShapesAndAnnotations,
    };
})();
