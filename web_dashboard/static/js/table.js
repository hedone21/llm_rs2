/**
 * Table — Filterable, sortable benchmark table.
 */

const Table = (() => {
    let _allProfiles = [];
    let _filtered = [];
    let _selected = new Set();
    let _sortKey = 'date';
    let _sortAsc = false;

    function init(profiles) {
        _allProfiles = profiles;
        _populateFilterOptions();
        _bindEvents();
        applyFilters();
    }

    function _populateFilterOptions() {
        // Prefill filter
        const prefills = [...new Set(_allProfiles.map(p => p.metadata.prefill_type).filter(Boolean))];
        const prefillSelect = document.getElementById('filter-prefill');
        prefillSelect.innerHTML = '<option value="">All</option>';
        for (const pf of prefills.sort()) {
            prefillSelect.innerHTML += `<option value="${pf}">${pf}</option>`;
        }

        // Tokens filter
        const tokens = [...new Set(_allProfiles.map(p => p.metadata.num_tokens).filter(Boolean))];
        const tokensSelect = document.getElementById('filter-tokens');
        tokensSelect.innerHTML = '<option value="">All</option>';
        for (const t of tokens.sort((a, b) => a - b)) {
            tokensSelect.innerHTML += `<option value="${t}">${t}</option>`;
        }
    }

    function _bindEvents() {
        document.getElementById('filter-backend').addEventListener('change', applyFilters);
        document.getElementById('filter-prefill').addEventListener('change', applyFilters);
        document.getElementById('filter-tokens').addEventListener('change', applyFilters);
        document.getElementById('filter-search').addEventListener('input', applyFilters);
        document.getElementById('btn-clear-filters').addEventListener('click', clearFilters);
        document.getElementById('select-all').addEventListener('change', _toggleSelectAll);

        // Table header sort
        document.querySelectorAll('#profile-table th[data-sort]').forEach(th => {
            th.addEventListener('click', () => {
                const key = th.dataset.sort;
                if (key === 'select') return;
                if (_sortKey === key) {
                    _sortAsc = !_sortAsc;
                } else {
                    _sortKey = key;
                    _sortAsc = true;
                }
                _renderTable();
            });
        });
    }

    function applyFilters() {
        const backend = document.getElementById('filter-backend').value;
        const prefill = document.getElementById('filter-prefill').value;
        const tokens = document.getElementById('filter-tokens').value;
        const search = document.getElementById('filter-search').value.toLowerCase();

        _filtered = _allProfiles.filter(p => {
            if (backend && p.metadata.backend !== backend) return false;
            if (prefill && p.metadata.prefill_type !== prefill) return false;
            if (tokens && String(p.metadata.num_tokens) !== tokens) return false;
            if (search && !p.id.toLowerCase().includes(search)) return false;
            return true;
        });

        _renderTable();
    }

    function clearFilters() {
        document.getElementById('filter-backend').value = '';
        document.getElementById('filter-prefill').value = '';
        document.getElementById('filter-tokens').value = '';
        document.getElementById('filter-search').value = '';
        applyFilters();
    }

    function _getEnvLabel(p) {
        const fg = p.metadata.foreground_app;
        return fg ? fg : 'Idle';
    }

    function _getSortValue(p) {
        switch (_sortKey) {
            case 'date': return p.metadata.date || '';
            case 'backend': return p.metadata.backend || '';
            case 'device': return p.metadata.device || '';
            case 'env': return _getEnvLabel(p);
            case 'prefill': return p.metadata.prefill_type || '';
            case 'eviction': return p.metadata.eviction_policy || '';
            case 'tokens': return p.metadata.num_tokens || 0;
            case 'ttft': return p.results.ttft_ms ?? 999999;
            case 'tbt': return p.results.tbt_ms ?? 999999;
            case 'tps': return p.results.tokens_per_sec ?? 0;
            case 'temp': return p.thermal.max_temp ?? 0;
            default: return '';
        }
    }

    function _renderTable() {
        const sorted = [..._filtered].sort((a, b) => {
            let va = _getSortValue(a);
            let vb = _getSortValue(b);
            if (typeof va === 'string') {
                return _sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
            }
            return _sortAsc ? va - vb : vb - va;
        });

        const tbody = document.getElementById('profile-table-body');
        tbody.innerHTML = '';

        for (const p of sorted) {
            const tr = document.createElement('tr');
            tr.classList.toggle('selected', _selected.has(p.id));
            tr.dataset.profileId = p.id;

            const date = p.metadata.date ? new Date(p.metadata.date).toLocaleDateString('ko-KR', {
                month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit'
            }) : '—';

            const backendBadge = p.metadata.backend
                ? `<span class="badge badge-${p.metadata.backend}">${p.metadata.backend}</span>`
                : '—';

            const envLabel = _getEnvLabel(p);
            const envBadge = envLabel === 'Idle'
                ? `<span class="badge badge-idle">Idle</span>`
                : `<span class="badge badge-fg">${envLabel}</span>`;

            let tempHtml = '—';
            if (p.thermal.slots && Object.keys(p.thermal.slots).length > 0) {
                // Render a compact badge for each slot: C:45° G:-- S:50° B:35°
                const s = p.thermal.slots;
                const badges = [];
                if (s.cpu) badges.push(`<span title="CPU Temp" class="t-badge c">C:${s.cpu.max.toFixed(0)}°</span>`);
                if (s.gpu) badges.push(`<span title="GPU Temp" class="t-badge g">G:${s.gpu.max.toFixed(0)}°</span>`);
                if (s.sys) badges.push(`<span title="SYS Temp" class="t-badge s">S:${s.sys.max.toFixed(0)}°</span>`);
                if (s.bat) badges.push(`<span title="BAT Temp" class="t-badge b">B:${s.bat.max.toFixed(0)}°</span>`);
                tempHtml = `<div class="temp-slots">${badges.join(' ')}</div>`;
            } else if (p.thermal.max_temp != null) {
                // Fallback for extremely old format
                tempHtml = `${p.thermal.max_temp.toFixed(0)}°C`;
            }

            tr.innerHTML = `
                <td><input type="checkbox" class="row-select" data-id="${p.id}" ${_selected.has(p.id) ? 'checked' : ''}></td>
                <td>${date}</td>
                <td>${backendBadge}</td>
                <td>${p.metadata.device || 'Unknown'}</td>
                <td>${envBadge}</td>
                <td>${p.metadata.prefill_type || '—'}</td>
                <td>${p.metadata.eviction_policy || '—'}</td>
                <td>${p.metadata.num_tokens ?? '—'}</td>
                <td>${p.results.ttft_ms != null ? p.results.ttft_ms.toFixed(1) : 'N/A'}</td>
                <td>${p.results.tbt_ms != null ? p.results.tbt_ms.toFixed(2) : 'N/A'}</td>
                <td>${p.results.tokens_per_sec != null ? p.results.tokens_per_sec.toFixed(1) : 'N/A'}</td>
                <td>${tempHtml}</td>
                <td><button class="btn btn-sm btn-secondary btn-view" data-id="${p.id}">View</button></td>
            `;

            tbody.appendChild(tr);
        }

        // Bind row events
        tbody.querySelectorAll('.row-select').forEach(cb => {
            cb.addEventListener('change', (e) => {
                if (e.target.checked) _selected.add(e.target.dataset.id);
                else _selected.delete(e.target.dataset.id);
                _updateSelectionUI();
            });
        });

        tbody.querySelectorAll('.btn-view').forEach(btn => {
            btn.addEventListener('click', () => {
                App.showDetail(btn.dataset.id);
            });
        });

        document.getElementById('table-status').textContent =
            `${sorted.length} of ${_allProfiles.length} profiles | ${_selected.size} selected`;

        _updateSelectionUI();
    }

    function _toggleSelectAll(e) {
        const checked = e.target.checked;
        _filtered.forEach(p => {
            if (checked) _selected.add(p.id);
            else _selected.delete(p.id);
        });
        _renderTable();
    }

    function _updateSelectionUI() {
        const n = _selected.size;
        document.getElementById('btn-compare-selected').disabled = n < 2;
        document.getElementById('btn-view-selected').disabled = n !== 1;
    }

    function getSelectedIds() {
        return [..._selected];
    }

    return { init, applyFilters, clearFilters, getSelectedIds };
})();
