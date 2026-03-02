const Todos = (() => {
    let _allTasks = [];

    async function load() {
        try {
            const resp = await fetch('/api/todos');
            if (!resp.ok) { _renderEmpty(); return; }
            const data = await resp.json();
            _allTasks = _flattenTasks(data.roles);
            _renderCards(data.summary);
            _renderProgressBar(data.summary);
            _renderFilters(data.roles);
            _renderTable(data.roles);
            _bindFilters(data.roles);
        } catch (e) {
            console.error('Todos load error:', e);
            _renderEmpty();
        }
    }

    function _flattenTasks(roles) {
        const tasks = [];
        for (const [role, info] of Object.entries(roles)) {
            for (const t of info.tasks) {
                tasks.push({ ...t, role });
            }
        }
        return tasks;
    }

    function _renderEmpty() {
        const el = document.getElementById('todos-cards');
        if (el) el.innerHTML = '<p style="color:var(--text-muted);padding:20px;">No TODO data available. Check <code>.agent/todos/</code> directory.</p>';
    }

    function _renderCards(summary) {
        const el = document.getElementById('todos-cards');
        if (!el) return;
        el.innerHTML = `
            <div class="summary-card">
                <div class="card-label">Total</div>
                <div class="card-value">${summary.total}</div>
            </div>
            <div class="summary-card">
                <div class="card-label">Done</div>
                <div class="card-value" style="color:var(--success)">${summary.done}</div>
            </div>
            <div class="summary-card">
                <div class="card-label">In Progress</div>
                <div class="card-value" style="color:var(--info)">${summary.in_progress}</div>
            </div>
            <div class="summary-card">
                <div class="card-label">Blocked</div>
                <div class="card-value" style="color:var(--warning)">${summary.blocked}</div>
            </div>
            <div class="summary-card">
                <div class="card-label">Completion</div>
                <div class="card-value">${summary.completion_pct}%</div>
            </div>
        `;
    }

    function _renderProgressBar(summary) {
        const el = document.getElementById('todos-progress');
        if (!el) return;
        const pct = summary.completion_pct;
        const color = pct >= 80 ? 'var(--success)' : pct >= 40 ? 'var(--warning)' : 'var(--accent)';
        el.innerHTML = `
            <div class="progress-bar-track">
                <div class="progress-bar-fill" style="width:${pct}%;background:${color}"></div>
            </div>
            <span class="progress-bar-label">${summary.done} / ${summary.total} tasks completed</span>
        `;
    }

    function _renderFilters(roles) {
        const roleSelect = document.getElementById('todo-filter-role');
        const statusSelect = document.getElementById('todo-filter-status');
        const sprintSelect = document.getElementById('todo-filter-sprint');
        const prioritySelect = document.getElementById('todo-filter-priority');
        if (!roleSelect) return;

        // Populate role options
        roleSelect.innerHTML = '<option value="">All Roles</option>';
        for (const role of Object.keys(roles)) {
            roleSelect.innerHTML += `<option value="${role}">${role}</option>`;
        }

        // Status options (fixed set)
        statusSelect.innerHTML = '<option value="">All Status</option>';
        for (const s of ['TODO', 'IN_PROGRESS', 'DONE', 'BLOCKED']) {
            statusSelect.innerHTML += `<option value="${s}">${s}</option>`;
        }

        // Sprint options
        const sprints = new Set(_allTasks.map(t => t.sprint).filter(Boolean));
        sprintSelect.innerHTML = '<option value="">All Sprints</option>';
        for (const s of sprints) {
            sprintSelect.innerHTML += `<option value="${s}">${s}</option>`;
        }

        // Priority options
        const priorities = new Set(_allTasks.map(t => t.priority).filter(Boolean));
        prioritySelect.innerHTML = '<option value="">All Priorities</option>';
        for (const p of [...priorities].sort()) {
            prioritySelect.innerHTML += `<option value="${p}">${p}</option>`;
        }
    }

    function _renderTable(roles, filters) {
        const tbody = document.getElementById('todos-table-body');
        if (!tbody) return;

        const f = filters || {};
        let html = '';

        for (const [role, info] of Object.entries(roles)) {
            if (f.role && f.role !== role) continue;

            const filtered = info.tasks.filter(t => {
                if (f.status && t.status.toUpperCase() !== f.status) return false;
                if (f.sprint && t.sprint !== f.sprint) return false;
                if (f.priority && t.priority !== f.priority) return false;
                return true;
            });

            if (filtered.length === 0) continue;

            html += `<tr class="role-header"><td colspan="7">${role}</td></tr>`;

            for (const t of filtered) {
                const statusClass = _statusBadgeClass(t.status);
                const prioClass = _prioBadgeClass(t.priority);
                html += `<tr>
                    <td><span class="badge ${prioClass}">${t.priority}</span></td>
                    <td><span class="badge ${statusClass}">${t.status}</span></td>
                    <td>${t.sprint || '—'}</td>
                    <td class="todo-title">${t.title}</td>
                    <td class="todo-deps">${t.dependencies || '—'}</td>
                    <td class="todo-desc">${t.description || '—'}</td>
                </tr>`;
            }
        }

        if (!html) {
            html = '<tr><td colspan="7" style="text-align:center;color:var(--text-muted);padding:20px;">No tasks match the current filters.</td></tr>';
        }

        tbody.innerHTML = html;
    }

    function _bindFilters(roles) {
        const ids = ['todo-filter-role', 'todo-filter-status', 'todo-filter-sprint', 'todo-filter-priority'];
        for (const id of ids) {
            const el = document.getElementById(id);
            if (el) {
                el.addEventListener('change', () => {
                    const filters = {
                        role: document.getElementById('todo-filter-role').value,
                        status: document.getElementById('todo-filter-status').value,
                        sprint: document.getElementById('todo-filter-sprint').value,
                        priority: document.getElementById('todo-filter-priority').value,
                    };
                    _renderTable(roles, filters);
                });
            }
        }
    }

    function _statusBadgeClass(status) {
        switch ((status || '').toUpperCase()) {
            case 'DONE': return 'badge-todo-done';
            case 'IN_PROGRESS': return 'badge-todo-progress';
            case 'BLOCKED': return 'badge-todo-blocked';
            default: return 'badge-todo-default';
        }
    }

    function _prioBadgeClass(priority) {
        switch (priority) {
            case 'P0': return 'badge-p0';
            case 'P1': return 'badge-p1';
            case 'P2': return 'badge-p2';
            case 'P3': return 'badge-p3';
            default: return 'badge-p3';
        }
    }

    return { load };
})();
