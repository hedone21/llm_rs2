from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
from web_dashboard.utils.result_reader import get_all_runs, get_run_detail
from web_dashboard.utils.runner import runner

app = FastAPI(title="LLM Benchmark Dashboard")

# Mount static files
app.mount("/static", StaticFiles(directory="web_dashboard/static"), name="static")
# Mount plots directory to serve generated plots
app.mount("/plots", StaticFiles(directory="results/plots"), name="plots")

class RunConfig(BaseModel):
    backend: str = "cpu"
    dry_run: bool = False
    skip_build: bool = True
    skip_push: bool = True

@app.get("/")
async def read_root():
    return FileResponse('web_dashboard/templates/index.html')

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/runs")
async def list_runs():
    return get_all_runs()

@app.get("/api/runs/{filename}")
async def get_run(filename: str):
    data = get_run_detail(filename)
    if not data:
        raise HTTPException(status_code=404, detail="Run not found")
    return data

@app.post("/api/run/start")
async def start_run(config: RunConfig):
    success = runner.start_run(config.dict())
    if not success:
        raise HTTPException(status_code=409, detail="A benchmark is already running")
    return {"status": "started"}

@app.post("/api/run/stop")
async def stop_run():
    runner.stop_run()
    return {"status": "stopping"}

@app.get("/api/run/status")
async def get_status():
    return runner.get_status()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
