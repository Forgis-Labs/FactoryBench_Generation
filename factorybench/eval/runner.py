from typing import Any, Dict, List
from datetime import datetime, timezone
import json
from pathlib import Path

from ..config import RUN_DIR, AZURE_PRICING
from ..adapters.base import ModelAdapter
from ..metrics.telemetry_literacy import score_sample, aggregate


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
) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)
    results: List[Dict[str, Any]] = []
    scores: List[Dict[str, Any]] = []
    # Prepare run skeleton with status for potential future incremental updates
    run_id = started.strftime("tl-%Y%m%dT%H%M%S")
    run: Dict[str, Any] = {
        "run_id": run_id,
        "stage": "telemetry_literacy",
        "model": model_name,
        "started_at": started.isoformat(),
        "dataset": dataset_meta,
        "results": [],
        "aggregate": {},
        "version": "0.1.0",
        "status": "running",
    }

    # Persist initial running state (enables future streaming/progress)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(RUN_DIR) / f"{run_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for s in samples:
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

    agg = aggregate(scores)
    # Cost computation (only azure pricing applies; mock will be zero)
    pricing = AZURE_PRICING.get(model_name, {})
    input_rate = pricing.get("input_per_1k", 0.0)
    output_rate = pricing.get("output_per_1k", 0.0)
    cost_input = (total_prompt_tokens / 1000.0) * input_rate
    cost_output = (total_completion_tokens / 1000.0) * output_rate
    cost_total = cost_input + cost_output

    agg["prompt_tokens_total"] = float(total_prompt_tokens)
    agg["completion_tokens_total"] = float(total_completion_tokens)
    agg["total_tokens"] = float(total_tokens)
    agg["cost_input"] = round(cost_input, 6)
    agg["cost_output"] = round(cost_output, 6)
    agg["cost_total"] = round(cost_total, 6)
    if agg.get("samples"):
        agg["cost_per_sample"] = round(cost_total / agg["samples"], 6)

    run["aggregate"] = agg
    run["results"] = results
    run["ended_at"] = datetime.now(timezone.utc).isoformat()
    run["status"] = "completed"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    return run
