from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any, List
from pathlib import Path
from datetime import datetime, timezone
import json

from ..config import (
    RUN_DIR,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    DATASETS,
    MODELS,
    MAX_COST_PER_RUN,
    MAX_COST_PER_DAY,
)
from ..stages import Stage, normalize_stage
from ..data.loader_tl import load_telemetry_literacy
from ..adapters.mock import MockAdapter
from ..adapters.azure_openai import AzureOpenAIAdapter
from ..eval.runner import run_telemetry_literacy
from ..viz.charts import generate_all_charts
from ..state import run_state

app = FastAPI(title="FactoryBench API", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHARTS_DIR = Path("charts")


class RunRequest(BaseModel):
    stage: Literal["telemetry_literacy"] = "telemetry_literacy"
    model: str = Field(default="mock", description="mock | azure:<deployment>")
    dataset_source: Literal["local", "hf"] = "local"
    dataset_id: Optional[str] = None  # Must be explicitly provided; no autodetection
    hf_slug: Optional[str] = None
    split: str = "train"
    limit: Optional[int] = 25
    fixture_path: str = "datasets/stage1.json"


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/runs")
def list_runs(
    model: Optional[List[str]] = Query(None),
    dataset: Optional[List[str]] = Query(None),
    stage: Optional[str] = None,
):
    """List runs with optional filtering by model, dataset ID, and stage."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(Path(RUN_DIR).glob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Apply filters
            if model and data.get("model") not in model:
                continue
            if stage and data.get("stage") != stage:
                continue
            if dataset:
                dataset_id = data.get("dataset", {}).get("dataset_id")
                if dataset_id not in dataset:
                    continue
            
            items.append(
                {
                    "run_id": data.get("run_id"),
                    "stage": data.get("stage"),
                    "model": data.get("model"),
                    "status": data.get("status", "completed"),
                    "aggregate": data.get("aggregate", {}),
                    "dataset": data.get("dataset", {}),
                }
            )
        except Exception:
            continue
    return {"items": items, "count": len(items)}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    p = Path(RUN_DIR) / f"{run_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/runs")
def create_run(req: RunRequest, background_tasks: BackgroundTasks):
    try:
        stage = normalize_stage(req.stage)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    if stage != Stage.telemetry_literacy:
        raise HTTPException(status_code=400, detail="Only telemetry_literacy supported in MVP")

    # Require explicit dataset_id and validate against registry
    dataset_id = req.dataset_id
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id required; provide a valid dataset id from registry")
    # Validate dataset id exists for the stage
    valid_ids = {d["id"] for d in DATASETS.get(req.stage, [])}
    if dataset_id not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Unknown dataset_id '{dataset_id}' for stage '{req.stage}'")
    
    # Generate run_id immediately
    run_id = datetime.now(timezone.utc).strftime("tl-%Y%m%dT%H%M%S")
    
    dataset_meta = {
        "source": req.dataset_source,
        "dataset_id": dataset_id,
        "hf_slug": req.hf_slug,
        "split": req.split,
        "limit": req.limit,
        "fixture_path": req.fixture_path,
    }
    
    # Create initial run file with running status
    initial_run = {
        "run_id": run_id,
        "stage": "telemetry_literacy",
        "model": req.model if req.model == "mock" else f"azure:{req.model.split(':', 1)[1] if ':' in req.model else req.model}",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_meta,
        "results": [],
        "aggregate": {},
        "version": "0.1.0",
        "status": "running",
        "loading_stage": "Loading dataset...",
    }
    
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(RUN_DIR) / f"{run_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(initial_run, f, indent=2)
    
    # Execute run in background
    background_tasks.add_task(_run_in_background, req, run_id, dataset_meta)
    
    # Return immediately with run_id
    return {"run_id": run_id, "status": "running", "message": "Run started in background"}


def _run_in_background(req: RunRequest, run_id: str, dataset_meta: Dict[str, Any]):
    """Execute benchmark run in background."""
    try:
        # Load dataset
        samples = load_telemetry_literacy(
            source=req.dataset_source,
            path=req.fixture_path,
            hf_slug=req.hf_slug,
            split=req.split,
            limit=req.limit,
        )
        if not samples:
            raise RuntimeError("No samples loaded")
        
        # Update loading stage
        out_path = Path(RUN_DIR) / f"{run_id}.json"
        if out_path.exists():
            with out_path.open("r", encoding="utf-8") as f:
                run = json.load(f)
            run["loading_stage"] = "Processing samples..."
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(run, f, indent=2)
        
        # Resolve adapter
        adapter, model_name = _resolve_adapter(req.model)
        
        # Run evaluation
        run_telemetry_literacy(samples, adapter, model_name, dataset_meta, run_id=run_id)
    except Exception as e:
        # Save error to run file
        out_path = Path(RUN_DIR) / f"{run_id}.json"
        if out_path.exists():
            with out_path.open("r", encoding="utf-8") as f:
                run = json.load(f)
            run["status"] = "failed"
            run["error"] = str(e)
            run["ended_at"] = datetime.now(timezone.utc).isoformat()
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(run, f, indent=2)
        run_state.complete_run(run_id, status="failed", error=str(e))


def _resolve_adapter(model: str):
    model = (model or "").strip()
    if model == "mock":
        return MockAdapter(), "mock"
    if model.startswith("azure:"):
        deployment = model.split(":", 1)[1]
        if not deployment:
            raise HTTPException(status_code=400, detail="Azure deployment name required: azure:<deployment>")
        missing = []
        if not AZURE_OPENAI_API_KEY:
            missing.append("AZURE_OPENAI_API_KEY")
        if not AZURE_OPENAI_ENDPOINT:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not AZURE_OPENAI_API_VERSION:
            missing.append("AZURE_OPENAI_API_VERSION")
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing Azure configuration vars: {', '.join(missing)}")
        return AzureOpenAIAdapter(deployment=deployment), f"azure:{deployment}"
    raise HTTPException(status_code=400, detail="Unknown model; try model=mock or azure:<deployment>")


@app.get("/charts/{chart_type}")
def get_chart(
    chart_type: str, 
    regenerate: bool = False,
    model: Optional[List[str]] = Query(None),
    dataset: Optional[List[str]] = Query(None),
):
    """Get a generated chart image with optional filtering."""
    # Always regenerate when filters are provided to ensure charts reflect current filter state
    should_regenerate = regenerate or model or dataset or not CHARTS_DIR.exists()
    
    if should_regenerate:
        generate_all_charts(RUN_DIR, CHARTS_DIR, model_filters=model, dataset_filters=dataset)
    
    chart_file = CHARTS_DIR / f"{chart_type}.png"
    if not chart_file.exists():
        raise HTTPException(status_code=404, detail=f"Chart {chart_type} not found")
    
    return FileResponse(chart_file, media_type="image/png")


@app.post("/charts/regenerate")
def regenerate_charts(
    model: Optional[List[str]] = Query(None),
    dataset: Optional[List[str]] = Query(None),
):
    """Regenerate all charts from current runs with optional filtering."""
    generate_all_charts(RUN_DIR, CHARTS_DIR, model_filters=model, dataset_filters=dataset)
    return {"status": "ok", "charts_dir": str(CHARTS_DIR), "filters": {"model": model, "dataset": dataset}}


@app.get("/metadata/models")
def get_models():
    """Get available models from registry and discovered from runs."""
    discovered_models = set()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for p in Path(RUN_DIR).glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                model = data.get("model")
                if model:
                    discovered_models.add(model)
        except Exception:
            continue
    
    # Combine registry models with discovered ones
    all_models = {m["id"]: m for m in MODELS}
    for model_id in discovered_models:
        if model_id not in all_models:
            all_models[model_id] = {
                "id": model_id,
                "name": model_id,
                "provider": "custom",
            }
    
    return {"models": list(all_models.values())}


@app.get("/metadata/datasets")
def get_datasets(stage: Optional[str] = None):
    """Get available datasets from registry."""
    if stage:
        return {"datasets": DATASETS.get(stage, [])}
    return {"datasets": DATASETS}


@app.get("/runs/{run_id}/progress")
def get_run_progress(run_id: str):
    """Get real-time progress for an active run."""
    progress = run_state.get_progress(run_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Run not found or not active")
    
    return {
        "run_id": progress.run_id,
        "total_samples": progress.total_samples,
        "processed_samples": progress.processed_samples,
        "current_cost": round(progress.current_cost, 6),
        "status": progress.status,
        "error": progress.error,
        "progress_percent": round((progress.processed_samples / progress.total_samples) * 100, 1) if progress.total_samples > 0 else 0,
        "cost_limit_per_run": MAX_COST_PER_RUN,
        "cost_limit_per_day": MAX_COST_PER_DAY,
        "daily_cost": round(run_state.get_daily_cost(), 6),
    }


@app.post("/runs/{run_id}/stop")
def stop_run(run_id: str):
    """Request a running benchmark to stop gracefully."""
    if run_state.request_stop(run_id):
        return {"status": "stop_requested", "run_id": run_id}
    raise HTTPException(status_code=404, detail="Run not found or not running")


@app.get("/metadata/cost-limits")
def get_cost_limits():
    """Get cost limits and current daily spend."""
    return {
        "max_cost_per_run": MAX_COST_PER_RUN,
        "max_cost_per_day": MAX_COST_PER_DAY,
        "daily_cost": round(run_state.get_daily_cost(), 6),
        "daily_remaining": round(MAX_COST_PER_DAY - run_state.get_daily_cost(), 6),
    }
