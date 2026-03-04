/**
 * Resilience — Resilience system visualization module.
 *
 * Shows signal types, strategy matrix, operating modes,
 * action reference, and conflict resolution rules.
 */

const Resilience = (() => {
    let _data = null;

    async function load() {
        try {
            const resp = await fetch('/api/resilience');
            if (!resp.ok) {
                _renderEmpty();
                return;
            }
            _data = await resp.json();
            _renderCards(_data.summary);
            _renderModes(_data.operating_modes);
            _renderMatrix(_data.strategies);
            _renderActions(_data.actions);
            _renderRules(_data.conflict_rules);
            _renderSignals(_data.signals);
        } catch (e) {
            console.error('Resilience load error:', e);
            _renderEmpty();
        }
    }

    function _renderEmpty() {
        const el = document.getElementById('resilience-cards');
        if (el) el.innerHTML = '<p style="color:var(--text-dim);padding:20px;">No resilience data available.</p>';
    }

    // ── Summary Cards ─────────────────────────────────

    function _renderCards(summary) {
        const el = document.getElementById('resilience-cards');
        if (!el) return;
        el.innerHTML = `
            <div class="summary-card">
                <div class="summary-label">Phase</div>
                <div class="summary-value" style="font-size:14px;">${summary.phase}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Signals</div>
                <div class="summary-value">${summary.signals}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Strategies</div>
                <div class="summary-value">${summary.strategies}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Actions</div>
                <div class="summary-value">${summary.actions}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Status</div>
                <div class="summary-value"><span class="badge badge-res-implemented">${summary.status}</span></div>
            </div>
        `;
    }

    // ── Operating Modes ───────────────────────────────

    function _renderModes(modes) {
        const el = document.getElementById('resilience-modes');
        if (!el) return;
        el.innerHTML = modes.map(m => `
            <div class="res-mode-card" style="border-left:4px solid ${m.color};">
                <div class="res-mode-name" style="color:${m.color};">${m.mode}</div>
                <div class="res-mode-cond">${m.condition}</div>
                <div class="res-mode-desc">${m.description}</div>
            </div>
        `).join('');
    }

    // ── Strategy Matrix ───────────────────────────────

    function _renderMatrix(strategies) {
        const tbody = document.getElementById('resilience-matrix-body');
        if (!tbody) return;

        const levels = ['Normal', 'Warning', 'Critical', 'Emergency'];
        const levelColors = {
            'Normal': '#22c55e',
            'Warning': '#f59e0b',
            'Critical': '#ef4444',
            'Emergency': '#6b7280',
        };

        let html = '';
        for (const [signal, levelMap] of Object.entries(strategies)) {
            for (let i = 0; i < levels.length; i++) {
                const level = levels[i];
                const actions = levelMap[level] || [];
                const actionHtml = actions.map(a => {
                    const cls = _actionBadgeClass(a.action);
                    const detail = a.detail ? ` <span class="res-detail">${a.detail}</span>` : '';
                    return `<span class="badge ${cls}">${a.action}</span>${detail}`;
                }).join(' ');

                html += '<tr>';
                if (i === 0) {
                    html += `<td rowspan="4" class="res-signal-cell">${signal}</td>`;
                }
                html += `<td><span style="color:${levelColors[level]};font-weight:500;">${level}</span></td>`;
                html += `<td>${actionHtml}</td>`;
                html += '</tr>';
            }
        }
        tbody.innerHTML = html;
    }

    // ── Action Reference ──────────────────────────────

    function _renderActions(actions) {
        const tbody = document.getElementById('resilience-actions-body');
        if (!tbody) return;
        tbody.innerHTML = actions.map(a => `
            <tr>
                <td><span class="badge ${_actionBadgeClass(a.name)}">${a.name}</span></td>
                <td>${a.target}</td>
                <td>${a.description}</td>
            </tr>
        `).join('');
    }

    // ── Conflict Rules ────────────────────────────────

    function _renderRules(rules) {
        const el = document.getElementById('resilience-rules');
        if (!el) return;
        el.innerHTML = '<ul>' + rules.map(r => `<li>${r}</li>`).join('') + '</ul>';
    }

    // ── D-Bus Signals ─────────────────────────────────

    function _renderSignals(signals) {
        const tbody = document.getElementById('resilience-signals-body');
        if (!tbody) return;
        tbody.innerHTML = signals.map(s => `
            <tr>
                <td><code>${s.dbus_member}</code></td>
                <td style="font-family:var(--font-mono);font-size:11px;">${s.params}</td>
                <td>${s.description}</td>
            </tr>
        `).join('');
    }

    // ── Helpers ───────────────────────────────────────

    function _actionBadgeClass(action) {
        const map = {
            'Evict': 'badge-res-evict',
            'SwitchBackend': 'badge-res-switch',
            'LimitTokens': 'badge-res-limit',
            'Throttle': 'badge-res-throttle',
            'Suspend': 'badge-res-suspend',
            'RejectNew': 'badge-res-reject',
            'RestoreDefaults': 'badge-res-restore',
        };
        return map[action] || 'badge-res-default';
    }

    return { load };
})();
