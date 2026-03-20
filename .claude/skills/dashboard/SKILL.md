---
name: dashboard
description: 벤치마크 웹 대시보드를 실행하고 관리한다. 프로파일링 결과 시각화 및 비교 분석.
allowed-tools: Bash, Read, Edit
---

# Dashboard

Flask 기반 벤치마크 시각화 대시보드. `http://localhost:5000`

## 실행

```bash
cd web_dashboard && source venv/bin/activate && python app.py
# 또는
cd web_dashboard && .venv/bin/python app.py
```

### 최초 설정
```bash
cd web_dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## 탭 구성

Overview, Table, Detail, Compare, Trends, Runner, Gates, Todos

## 새 필드 추가

`web_dashboard/backend/schema_registry.py`의 `TIMESERIES_FIELDS`에 항목 추가만 하면 프론트엔드가 자동 인식:

```python
"power_mw": {
    "label": "Power Draw", "unit": "mW", "color": "#f59e0b",
    "group": "power", "chart": "line", "axis_label": "Power (mW)",
},
```

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/profiles` | 전체 프로파일 요약 |
| GET | `/api/profiles/<id>` | 상세 프로파일 + 시계열 |
| GET | `/api/compare?ids=a,b` | 다중 프로파일 비교 |
| POST | `/api/benchmark/run` | 벤치마크 실행 |
| GET | `/api/todos` | TODO 목록 조회 |

## 트러블슈팅

| 문제 | 해결 |
|------|------|
| `ModuleNotFoundError: flask` | `source venv/bin/activate` 먼저 |
| Port 5000 사용 중 | `lsof -i:5000` → kill |
| 프로파일 없음 | `results/data/`에 `.json` 파일 확인 |
