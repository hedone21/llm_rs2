/**
 * App — Main application controller.
 *
 * Handles tab routing, data loading, and coordination between modules.
 */

const App = (() => {
    let _profiles = [];
    let _currentTab = 'overview';

    // ── Init ────────────────────────────────────────────────

    async function init() {
        _bindNavigation();
        _bindTableActions();
        Runner.init();

        await reload();

        // Trends controls
        document.getElementById('trends-backend').addEventListener('change', _renderTrends);
        document.getElementById('trends-metric').addEventListener('change', _renderTrends);
    }

    async function reload() {
        try {
            const resp = await fetch('/api/profiles');
            const data = await resp.json();
            _profiles = data.profiles || [];

            document.getElementById('profile-count-badge').textContent = `${_profiles.length} profiles`;

            Table.init(_profiles);
            _renderOverview();
            _renderTrends();
        } catch (err) {
            console.error('Failed to load profiles:', err);
        }
    }

    // ── Navigation ──────────────────────────────────────────

    function _bindNavigation() {
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                switchTab(btn.dataset.tab);
            });
        });
    }

    function switchTab(tabName) {
        _currentTab = tabName;

        // Update nav buttons
        document.querySelectorAll('.nav-btn').forEach(b => {
            b.classList.toggle('active', b.dataset.tab === tabName);
        });

        // Update panels
        document.querySelectorAll('.tab-panel').forEach(p => {
            p.classList.toggle('active', p.id === `tab-${tabName}`);
        });

        // Trigger resize for Plotly charts
        setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
    }

    // ── Table Actions ───────────────────────────────────────

    function _bindTableActions() {
        document.getElementById('btn-compare-selected').addEventListener('click', () => {
            const ids = Table.getSelectedIds();
            if (ids.length >= 2) {
                switchTab('compare');
                Compare.loadAndRender(ids);
            }
        });

        document.getElementById('btn-view-selected').addEventListener('click', () => {
            const ids = Table.getSelectedIds();
            if (ids.length === 1) {
                showDetail(ids[0]);
            }
        });
    }

    // ── Detail View ─────────────────────────────────────────

    async function showDetail(profileId) {
        switchTab('detail');

        const headerEl = document.getElementById('detail-header');
        const metaEl = document.getElementById('detail-meta');
        const chartsEl = document.getElementById('detail-charts');

        headerEl.innerHTML = '<h2>⏳ Loading...</h2>';
        metaEl.innerHTML = '';
        chartsEl.innerHTML = '';

        try {
            const resp = await fetch(`/api/profiles/${profileId}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const profile = await resp.json();

            // Header
            const meta = profile.metadata || {};
            const results = profile.results || {};

            headerEl.innerHTML = `
                <h2>${meta.model || 'Unknown Model'}
                    <span class="badge badge-${meta.backend || ''}">${meta.backend || '?'}</span>
                </h2>
                <p style="color:var(--text-secondary);font-size:13px;">
                    ${meta.date || ''} · ${meta.prefill_type || ''} · ${meta.num_tokens || '?'} tokens
                </p>
            `;

            // Meta cards
            const metaItems = [
                { label: 'TTFT', value: results.ttft_ms != null ? `${results.ttft_ms.toFixed(1)}` : 'N/A', unit: 'ms' },
                { label: 'Avg TBT', value: results.tbt_ms != null ? `${results.tbt_ms.toFixed(2)}` : 'N/A', unit: 'ms' },
                { label: 'Tokens/sec', value: results.tokens_per_sec != null ? `${results.tokens_per_sec.toFixed(1)}` : 'N/A', unit: 'tok/s' },
                { label: 'Start Temp', value: profile.thermal?.start_temp ?? '—', unit: '°C' },
                { label: 'Max Temp', value: profile.thermal?.max_temp ?? '—', unit: '°C' },
                { label: 'Baseline Memory', value: profile.baseline?.avg_memory_used_mb != null ? `${profile.baseline.avg_memory_used_mb.toFixed(0)}` : '—', unit: 'MB' },
            ];

            metaEl.innerHTML = metaItems.map(item => `
                <div class="meta-item">
                    <div class="meta-label">${item.label}</div>
                    <div class="meta-value">${item.value} <span class="meta-unit">${item.unit}</span></div>
                </div>
            `).join('');

            // Charts
            Presenter.renderDetailCharts('detail-charts', profile);

        } catch (err) {
            headerEl.innerHTML = `<p class="placeholder-text">Error: ${err.message}</p>`;
        }
    }

    // ── Overview ─────────────────────────────────────────────

    function _renderOverview() {
        const valid = _profiles.filter(p => p.results.tokens_per_sec != null);
        const cpu = valid.filter(p => p.metadata.backend === 'cpu');
        const opencl = valid.filter(p => p.metadata.backend === 'opencl');

        // Summary cards
        const latestDate = _profiles.length > 0
            ? _profiles.reduce((a, b) => (a.metadata.date || '') > (b.metadata.date || '') ? a : b).metadata.date
            : null;
        const latestStr = latestDate ? new Date(latestDate).toLocaleDateString('ko-KR') : '—';

        const avgTps = valid.length > 0
            ? (valid.reduce((s, p) => s + p.results.tokens_per_sec, 0) / valid.length).toFixed(1)
            : '—';

        document.getElementById('overview-cards').innerHTML = `
            <div class="summary-card">
                <div class="card-label">Total Benchmarks</div>
                <div class="card-value">${_profiles.length}</div>
            </div>
            <div class="summary-card">
                <div class="card-label">CPU Runs</div>
                <div class="card-value" style="color:#a78bfa">${cpu.length}</div>
            </div>
            <div class="summary-card">
                <div class="card-label">OpenCL Runs</div>
                <div class="card-value" style="color:#22d3ee">${opencl.length}</div>
            </div>
            <div class="summary-card">
                <div class="card-label">Avg Tokens/sec</div>
                <div class="card-value">${avgTps}</div>
            </div>
            <div class="summary-card">
                <div class="card-label">Latest Run</div>
                <div class="card-value" style="font-size:16px;">${latestStr}</div>
            </div>
        `;

        // Recent benchmarks list
        const recent = [..._profiles]
            .sort((a, b) => (b.metadata.date || '').localeCompare(a.metadata.date || ''))
            .slice(0, 8);

        document.getElementById('recent-list').innerHTML = recent.map(p => {
            const tps = p.results.tokens_per_sec != null ? `${p.results.tokens_per_sec.toFixed(1)} tok/s` : 'N/A';
            const badge = `<span class="badge badge-${p.metadata.backend || ''}">${p.metadata.backend || '?'}</span>`;
            return `<div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--border);font-size:12px;">
                <span>${badge} ${p.metadata.prefill_type || '?'} / ${p.metadata.num_tokens || '?'}tok</span>
                <span style="font-family:var(--font-mono);color:var(--text-secondary)">${tps}</span>
            </div>`;
        }).join('');

        // Charts
        Presenter.renderOverviewCharts(_profiles);
    }

    // ── Trends ──────────────────────────────────────────────

    function _renderTrends() {
        const backend = document.getElementById('trends-backend').value;
        const metric = document.getElementById('trends-metric').value;
        Presenter.renderTrendsChart(_profiles, metric, backend || null);
    }

    // Public API
    return { init, reload, showDetail, switchTab };
})();

// Boot
document.addEventListener('DOMContentLoaded', () => App.init());
