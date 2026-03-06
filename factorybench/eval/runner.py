from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import json
from pathlib import Path

from ..config import RUN_DIR, AZURE_PRICING, MAX_COST_PER_RUN, MAX_COST_PER_DAY
from ..adapters.base import ModelAdapter
from ..metrics.telemetry_literacy import score_sample, aggregate
from ..state import run_state


def build_prompt(sample: Dict[str, Any]) -> str:
    values = sample.get("values", [])
    timestamps = sample.get("timestamps", [])
    
    return (
        "You are given a numeric time series with timestamps. Compute and return only these values:\n"
        "mean=<float> min=<float> max=<float>\n"
        f"Timestamps: {timestamps}\n"
        f"Values: {values}\n"
        "Output format: mean=<float> min=<float> max=<float>"
    )


def run_telemetry_literacy(
    samples: List[Dict[str, Any]],
    adapter: ModelAdapter,
    model_name: str,
    dataset_meta: Dict[str, Any],
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)
    if run_id is None:
        run_id = started.strftime("tl-%Y%m%dT%H%M%S")
    
    # Initialize progress tracking
    progress = run_state.start_run(run_id, total_samples=len(samples))
    
    # Check daily cost limit before starting
    daily_cost = run_state.get_daily_cost()
    if daily_cost >= MAX_COST_PER_DAY:
        run_state.complete_run(run_id, status="failed", error=f"Daily cost limit reached: ${daily_cost:.2f} >= ${MAX_COST_PER_DAY}")
        raise RuntimeError(f"Daily cost limit reached: ${daily_cost:.2f}. Maximum allowed: ${MAX_COST_PER_DAY}/day")
    
    results: List[Dict[str, Any]] = []
    scores: List[Dict[str, Any]] = []
    
    # Persist initial running state
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(RUN_DIR) / f"{run_id}.json"
    
    # Load existing run if it exists (from API initial creation), otherwise use skeleton
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            run = json.load(f)
    else:
        run = {
            "run_id": run_id,
            "stage": "telemetry_literacy",
            "model": model_name,
            "version": "0.1.0",
        }
    
    # Update with fresh data and remove loading_stage
    run.update({
        "started_at": started.isoformat(),
        "dataset": dataset_meta,
        "results": [],
        "aggregate": {},
        "status": "running",
    })
    # Explicitly remove loading_stage
    run.pop("loading_stage", None)
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    cost_total = 0.0
    
    # Get pricing info
    pricing = AZURE_PRICING.get(model_name, {})
    input_rate = pricing.get("input_per_1k", 0.0)
    output_rate = pricing.get("output_per_1k", 0.0)

    try:
        for idx, s in enumerate(samples):
            # Check if stop was requested
            if run_state.should_stop(run_id):
                run["status"] = "stopped"
                break
            
            # Check per-run cost limit
            if cost_total >= MAX_COST_PER_RUN:
                run["status"] = "stopped"
                run["stop_reason"] = f"Cost limit reached: ${cost_total:.4f} >= ${MAX_COST_PER_RUN}"
                break
            
            # Check daily cost limit (including current run)
            if daily_cost + cost_total >= MAX_COST_PER_DAY:
                run["status"] = "stopped"
                run["stop_reason"] = f"Daily cost limit reached: ${daily_cost + cost_total:.2f} >= ${MAX_COST_PER_DAY}"
                break
            
            prompt = build_prompt(s)
            gen = adapter.generate(prompt)
            pred_text = gen.get("text", "")
            usage = gen.get("usage", {}) or {}
            prompt_tokens = usage.get("prompt_tokens") or 0
            completion_tokens = usage.get("completion_tokens") or 0
            all_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
            
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += all_tokens
            
            # Update cost after each sample
            cost_input = (total_prompt_tokens / 1000.0) * input_rate
            cost_output = (total_completion_tokens / 1000.0) * output_rate
            cost_total = cost_input + cost_output

            sc = score_sample(s, pred_text)
            scores.append(sc)
            
            result_item = {
                "id": s.get("id"),
                "values": s.get("values"),
                "timestamps": s.get("timestamps"),
                "domain": s.get("domain"),
                "subtype": s.get("subtype"),
                "statistics": s.get("statistics"),
                "prediction_text": pred_text,
                "metrics": sc,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": all_tokens,
                },
            }
            
            results.append(result_item)
            
            # Update progress
            run_state.update_progress(run_id, processed_samples=idx + 1, current_cost=cost_total)
            
            # Save incremental progress after every sample
            run["results"] = results
            run["aggregate"] = _compute_aggregate(scores, total_prompt_tokens, total_completion_tokens, total_tokens, cost_total, model_name)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(run, f, indent=2)
        
        # Mark as completed if we processed all samples
        if len(results) == len(samples) and run["status"] == "running":
            run["status"] = "completed"
            
    except Exception as e:
        run["status"] = "failed"
        run["error"] = str(e)
        run_state.complete_run(run_id, status="failed", error=str(e))
        raise
    finally:
        # Final aggregate and save
        agg = _compute_aggregate(scores, total_prompt_tokens, total_completion_tokens, total_tokens, cost_total, model_name)
        run["aggregate"] = agg
        run["results"] = results
        run["ended_at"] = datetime.now(timezone.utc).isoformat()
        
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(run, f, indent=2)
        
        # Mark run as complete in state manager
        run_state.complete_run(run_id, status=run["status"])
    
    return run


def _compute_aggregate(
    scores: List[Dict[str, Any]],
    total_prompt_tokens: int,
    total_completion_tokens: int,
    total_tokens: int,
    cost_total: float,
    model_name: str
) -> Dict[str, Any]:
    """Compute aggregate metrics from scores and usage."""
    agg = aggregate(scores)
    
    agg["prompt_tokens_total"] = float(total_prompt_tokens)
    agg["completion_tokens_total"] = float(total_completion_tokens)
    agg["total_tokens"] = float(total_tokens)
    agg["cost_input"] = round((total_prompt_tokens / 1000.0) * AZURE_PRICING.get(model_name, {}).get("input_per_1k", 0.0), 6)
    agg["cost_output"] = round((total_completion_tokens / 1000.0) * AZURE_PRICING.get(model_name, {}).get("output_per_1k", 0.0), 6)
    agg["cost_total"] = round(cost_total, 6)
    
    if agg.get("samples"):
        agg["cost_per_sample"] = round(cost_total / agg["samples"], 6)
    
    return agg
