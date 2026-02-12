---
name: dashboard
description: Run and extend the web dashboard for benchmark visualization and analysis.
---

# Dashboard Skill

Use this skill when running, extending, or debugging the `web_dashboard` application.

## 1. Running the Dashboard

### Quick Start
```bash
cd web_dashboard
source venv/bin/activate
python app.py
# → http://localhost:5000
```

### First-Time Setup
If the virtual environment does not exist yet:
```bash
cd web_dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## 2. Architecture Overview

```
web_dashboard/
├── app.py                     # Flask entry (port 5000)
├── backend/
│   ├── api.py                 # 7 REST endpoints
│   ├── parser.py              # Flexible JSON profile parser
│   ├── schema_registry.py     # Field metadata registry (★ KEY FILE)
│   └── runner.py              # android_profile.py subprocess wrapper
├── static/
│   ├── css/style.css          # Dark research theme
│   └── js/
│       ├── app.js             # Tab routing + data loading
│       ├── presenter.js       # Schema-driven Plotly chart builder
│       ├── table.js           # Filterable/sortable table
│       ├── compare.js         # Multi-profile overlay comparison
│       └── runner.js          # Benchmark execution panel
├── templates/index.html       # SPA shell (6 tabs)
├── requirements.txt           # Python deps (flask)
└── venv/                      # Python virtual environment
```

### Data Flow
1. `parser.py` reads `results/data/*.json` → normalizes to common format
2. `schema_registry.py` provides field metadata (label, unit, color, group, chart type)
3. `api.py` serves normalized data + field descriptors via REST
4. `presenter.js` reads field descriptors → builds Plotly subplots dynamically

## 3. Extending: Adding New Timeseries Fields

> [!IMPORTANT]
> When the JSON schema gains a new timeseries field (e.g., `power_mw`), only **one file** needs editing.

### Step 1: Edit `backend/schema_registry.py`

Add one entry to `TIMESERIES_FIELDS`:

```python
"power_mw": {
    "label": "Power Draw",
    "unit": "mW",
    "color": "#f59e0b",
    "group": "power",          # same group → same subplot
    "chart": "line",           # or "multi_line" for arrays
    "axis_label": "Power (mW)",
    # Optional:
    # "transform": "custom_fn",
    # "style": "dash",
},
```

### Step 2 (Optional): Add transforms

If the raw value needs conversion, add to `TRANSFORMS` dict:

```python
TRANSFORMS = {
    "khz_to_ghz": lambda v: round(v / 1_000_000, 3),
    "hz_to_mhz":  lambda v: round(v / 1_000_000, 1),
    "mw_to_w":    lambda v: round(v / 1000, 2),       # ← new
}
```

Then reference it in the field entry: `"transform": "mw_to_w"`.

**No frontend changes are required — the presenter auto-discovers fields.**

## 4. Extending: Adding New Tabs/Views

1. Add a `<section id="tab-newname" class="tab-panel">` in `templates/index.html`
2. Add a `<button class="nav-btn" data-tab="newname">` to the nav bar
3. Create `static/js/newname.js` with an IIFE module
4. Include the script in `index.html` before `app.js`
5. Hook into `App.init()` in `app.js` if needed

## 5. API Reference

| Method | Path | Returns |
|:---|:---|:---|
| `GET` | `/api/profiles` | All profile summaries |
| `GET` | `/api/profiles/<id>` | Full profile + timeseries + field descriptors |
| `GET` | `/api/profiles/<id>/timeseries` | Timeseries only + field descriptors |
| `GET` | `/api/schema` | Schema Registry contents |
| `GET` | `/api/compare?ids=a,b` | Multiple full profiles |
| `POST` | `/api/benchmark/run` | Start benchmark `{backend, prefill_type, num_tokens}` |
| `GET` | `/api/benchmark/status` | Execution status + log |

## 6. Troubleshooting

| Problem | Solution |
|:---|:---|
| `ModuleNotFoundError: flask` | `source venv/bin/activate` first |
| Port 5000 in use | `lsof -i:5000` then kill, or change port in `app.py` |
| No profiles loaded | Check `results/data/` has `.json` files |
| Charts don't render | Check browser console; ensure CDN (plot.ly) is reachable |
