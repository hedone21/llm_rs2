document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentView = 'dashboard';
    let pollingInterval = null;
    let isRunning = false;

    // Elements
    const dashboardView = document.getElementById('dashboard-view');
    const detailView = document.getElementById('detail-view');
    const tableBody = document.getElementById('runs-table-body');
    const totalRunsEl = document.getElementById('total-runs');
    const latestRunDateEl = document.getElementById('latest-run-date');
    const refreshBtn = document.getElementById('refresh-btn');
    const newRunBtn = document.getElementById('new-run-btn');
    const runModal = document.getElementById('run-modal');
    const closeModal = document.querySelector('.close-modal');
    const runForm = document.getElementById('run-form');
    const terminal = document.getElementById('run-terminal');
    const terminalOutput = document.getElementById('terminal-output');
    const backBtn = document.querySelector('.back-btn');

    // Detail Elements
    const detailTitle = document.getElementById('detail-title');
    const detailJson = document.getElementById('detail-json');
    const detailMetricsContainer = document.getElementById('detail-metrics');
    let metricsChartInstance = null;

    // Init
    fetchRuns();
    checkRunStatus();

    // Event Listeners
    refreshBtn.addEventListener('click', fetchRuns);

    newRunBtn.addEventListener('click', () => {
        runModal.classList.remove('hidden');
    });

    closeModal.addEventListener('click', () => {
        if (!isRunning) {
            runModal.classList.add('hidden');
        } else {
            // If running, just hide the modal but keep polling?
            // For now, simple hide.
            runModal.classList.add('hidden');
        }
    });

    backBtn.addEventListener('click', () => {
        showView('dashboard');
    });

    runForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(runForm);
        const config = {
            backend: formData.get('backend'),
            dry_run: formData.get('dry_run') === 'on',
            skip_build: formData.get('skip_build') === 'on',
            skip_push: formData.get('skip_push') === 'on'
        };

        try {
            const res = await fetch('/api/run/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            if (res.ok) {
                isRunning = true;
                terminal.classList.remove('hidden');
                runForm.querySelector('button').disabled = true;
                startPolling();
            } else {
                const err = await res.json();
                alert(`Failed to start: ${err.detail}`);
            }
        } catch (error) {
            console.error(error);
            alert('Error starting run');
        }
    });

    // Functions
    function showView(view) {
        if (view === 'dashboard') {
            dashboardView.classList.remove('hidden');
            detailView.classList.add('hidden');
        } else {
            dashboardView.classList.add('hidden');
            detailView.classList.remove('hidden');
        }
        currentView = view;
    }

    async function fetchRuns() {
        try {
            const res = await fetch('/api/runs');
            const runs = await res.json();
            renderTable(runs);
            updateStats(runs);
        } catch (error) {
            console.error('Error fetching runs:', error);
        }
    }

    function renderTable(runs) {
        tableBody.innerHTML = '';
        runs.forEach(run => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${run.date}</td>
                <td>${run.model}</td>
                <td>${run.backend}</td>
                <td>${run.input_type}</td>
                <td>${run.num_tokens}</td>
                <td>${run.ttft_ms}</td>
                <td>${run.tbt_ms}</td>
                <td>${run.tokens_per_sec}</td>
                <td><button class="view-btn" data-filename="${run.filename}">View</button></td>
            `;
            tableBody.appendChild(tr);
        });

        // Add event listeners for view buttons
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const filename = e.target.getAttribute('data-filename');
                loadRunDetail(filename);
            });
        });
    }

    function updateStats(runs) {
        totalRunsEl.textContent = runs.length;
        if (runs.length > 0) {
            latestRunDateEl.textContent = runs[0].date;
        }
    }

    async function loadRunDetail(filename) {
        try {
            const res = await fetch(`/api/runs/${filename}`);
            const data = await res.json();

            detailTitle.textContent = `Run Details: ${filename}`;
            detailJson.textContent = JSON.stringify(data, null, 2);

            renderDetailMetrics(data);
            renderChart(data);

            showView('detail');
        } catch (error) {
            console.error('Error loading detail:', error);
            alert('Failed to load run details');
        }
    }

    function renderDetailMetrics(data) {
        const results = data.benchmark_results || {};
        const baseline = data.baseline || {};

        const metrics = [
            { label: 'TTFT', value: `${results.ttft_ms || '-'} ms` },
            { label: 'TBT', value: `${results.tbt_ms || '-'} ms` },
            { label: 'Tokens/Sec', value: `${results.tokens_per_sec || '-'} t/s` },
            { label: 'Avg Pkg Temp', value: `${baseline.avg_pkg_temp_c || '-'} °C` },
            { label: 'Max Pkg Temp', value: `${baseline.max_pkg_temp_c || '-'} °C` },
            { label: 'Memory Used', value: `${baseline.avg_memory_used_mb || '-'} MB` },
        ];

        detailMetricsContainer.innerHTML = metrics.map(m => `
            <div class="card">
                <h3>${m.label}</h3>
                <span>${m.value}</span>
            </div>
        `).join('');

        // Render Events Timeline
        const events = data.events || [];
        if (events.length > 0) {
            const startTime = new Date(events[0].timestamp).getTime();
            const timelineHtml = events.map(e => {
                const time = new Date(e.timestamp).getTime();
                const diff = (time - startTime) / 1000; // seconds
                return `
                    <div class="event-item">
                        <span class="event-time">+${diff.toFixed(3)}s</span>
                        <span class="event-name">${e.name}</span>
                    </div>
                `;
            }).join('');

            // Append timeline to metrics container or a new container
            // Let's create a new card for timeline if not exists
            let timelineContainer = document.getElementById('detail-timeline');
            if (!timelineContainer) {
                timelineContainer = document.createElement('div');
                timelineContainer.id = 'detail-timeline';
                timelineContainer.className = 'card timeline-container';
                timelineContainer.innerHTML = '<h3>Execution Timeline</h3><div class="timeline-list"></div>';
                detailMetricsContainer.parentNode.insertBefore(timelineContainer, detailMetricsContainer.nextSibling);
            }
            timelineContainer.querySelector('.timeline-list').innerHTML = timelineHtml;
        } else {
            const existingTimeline = document.getElementById('detail-timeline');
            if (existingTimeline) existingTimeline.remove();
        }
    }

    function renderChart(data) {
        const timeseries = data.timeseries || [];
        const ctx = document.getElementById('metricsChart').getContext('2d');

        if (metricsChartInstance) {
            metricsChartInstance.destroy();
        }

        if (timeseries.length === 0) {
            // Handle empty data case
            return;
        }

        const labels = timeseries.map((_, i) => i); // Just index or timestamp if available
        const temps = timeseries.map(d => d.temp_c);
        const mems = timeseries.map(d => d.mem_used_mb);
        const cpuLoads = timeseries.map(d => d.cpu_load_percent || 0);
        const gpuLoads = timeseries.map(d => d.gpu_load_percent || 0);

        // Optional Process Metrics (if present)
        const processCpu = timeseries.map(d => d.process_cpu_percent || null); // Use null to skip drawing if missing
        const processMem = timeseries.map(d => d.process_mem_mb || null);

        const datasets = [
            {
                label: 'Temperature (°C)',
                data: temps,
                borderColor: '#ff6384',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                yAxisID: 'y_temp',
                tension: 0.4
            },
            {
                label: 'CPU Load (%)',
                data: cpuLoads,
                borderColor: '#4bc0c0',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                yAxisID: 'y_load',
                borderDash: [5, 5],
                tension: 0.4
            },
            {
                label: 'GPU Load (%)',
                data: gpuLoads,
                borderColor: '#9966ff',
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                yAxisID: 'y_load',
                borderDash: [2, 2],
                tension: 0.4
            },
            {
                label: 'Global Memory (MB)',
                data: mems,
                borderColor: '#36a2eb',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                yAxisID: 'y_mem',
                tension: 0.4
            }
        ];

        // Add Process Metrics if any data point is valid
        if (processCpu.some(v => v !== null)) {
            datasets.push({
                label: 'Process CPU (%)',
                data: processCpu,
                borderColor: '#ffcd56', // Yellow
                backgroundColor: 'rgba(255, 205, 86, 0.2)',
                yAxisID: 'y_load',
                tension: 0.4
            });
        }
        if (processMem.some(v => v !== null)) {
            datasets.push({
                label: 'Process Mem (MB)',
                data: processMem,
                borderColor: '#ff9f40', // Orange
                backgroundColor: 'rgba(255, 159, 64, 0.2)',
                yAxisID: 'y_mem',
                borderDash: [5, 5],
                tension: 0.4
            });
        }

        metricsChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            pointStyle: 'rectRounded',
                            boxWidth: 10
                        },
                        onClick: (e, legendItem, legend) => {
                            const index = legendItem.datasetIndex;
                            const ci = legend.chart;
                            if (ci.isDatasetVisible(index)) {
                                ci.hide(index);
                                legendItem.hidden = true;
                            } else {
                                ci.show(index);
                                legendItem.hidden = false;
                            }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    },
                    // Custom plugin definition inline or external
                    annotation: {
                        // Placeholder if using chartjs-plugin-annotation, but we'll use a custom one below
                    }
                },
                scales: {
                    y_temp: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: { display: true, text: 'Temperature (°C)' },
                        suggestedMax: 80
                    },
                    y_load: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: { display: true, text: 'Load (%)' },
                        grid: { drawOnChartArea: false }, // Avoid grid overlap
                        suggestedMax: 100
                    },
                    y_mem: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: { display: true, text: 'Memory (MB)' },
                        grid: { drawOnChartArea: false },
                    }
                }
            },
            plugins: [{
                id: 'eventMarkers',
                afterDraw: (chart) => {
                    const ctx = chart.ctx;
                    const xAxis = chart.scales.x;
                    const yAxis = chart.scales.y_temp;
                    const top = yAxis.top;
                    const bottom = yAxis.bottom;

                    if (!data.events) return;

                    // Helper to correct timestamp (some legacy data might differ)
                    const getTs = (ts) => new Date(ts).getTime();

                    // Map events to x-axis positions
                    // timeseries data has timestamps? We used index as label.
                    // We need to match event time to index.
                    const startTime = getTs(timeseries[0].timestamp);
                    const endTime = getTs(timeseries[timeseries.length - 1].timestamp);
                    const duration = endTime - startTime;

                    data.events.forEach(event => {
                        const eventName = event.name.replace('Start', ''); // Shorten name
                        const eventTime = getTs(event.timestamp);

                        // Find nearest index
                        // Simple ratio if linear, but timeseries might have gaps/fluctuations.
                        // Best is to find index where timeseries[i].timestamp >= event.timestamp
                        let index = timeseries.findIndex(d => getTs(d.timestamp) >= eventTime);
                        if (index === -1) {
                            if (eventTime > endTime) index = timeseries.length - 1;
                            else index = 0;
                        }

                        const x = xAxis.getPixelForValue(index);

                        // Draw Line
                        ctx.save();
                        ctx.beginPath();
                        ctx.moveTo(x, top);
                        ctx.lineTo(x, bottom);
                        ctx.lineWidth = 2;
                        ctx.strokeStyle = 'rgba(255, 99, 132, 0.8)'; // Red-ish
                        if (eventName === 'Decoding') ctx.strokeStyle = 'rgba(75, 192, 192, 0.8)'; // Green-ish
                        if (eventName === 'Prefill') ctx.strokeStyle = 'rgba(255, 205, 86, 0.8)'; // Yellow-ish
                        ctx.setLineDash([5, 5]);
                        ctx.stroke();

                        // Draw Label
                        ctx.fillStyle = ctx.strokeStyle;
                        ctx.textAlign = 'center';
                        ctx.font = 'bold 12px sans-serif';
                        ctx.fillText(eventName, x, top - 5);
                        ctx.restore();
                    });
                }
            }]
        });
    }

    function startPolling() {
        if (pollingInterval) clearInterval(pollingInterval);

        pollingInterval = setInterval(async () => {
            await checkRunStatus();
        }, 1000);
    }

    async function checkRunStatus() {
        try {
            const res = await fetch('/api/run/status');
            const statusData = await res.json();

            if (statusData.status === 'RUNNING') {
                isRunning = true;
                terminal.classList.remove('hidden');
                terminalOutput.textContent = statusData.logs;
                // Auto scroll to bottom
                terminal.scrollTop = terminal.scrollHeight;

                // Ensure modal is open if running
                if (runModal.classList.contains('hidden') && currentView === 'dashboard') {
                    // Optional: don't force open, maybe show a badge
                }
            } else {
                isRunning = false;
                if (pollingInterval) {
                    clearInterval(pollingInterval);
                    pollingInterval = null;
                }

                terminalOutput.textContent = statusData.logs; // Final logs
                runForm.querySelector('button').disabled = false;

                if (statusData.status === 'COMPLETED') {
                    fetchRuns(); // Refresh list
                }
            }
        } catch (error) {
            console.error('Error checking status:', error);
        }
    }
});
