from typing import Dict, Any, List
import math


def parse_prediction(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    tokens = text.replace(",", " ").split()
    for tok in tokens:
        if "=" in tok:
            k, v = tok.split("=", 1)
            k = k.strip().lower()
            try:
                out[k] = float(v)
            except Exception:
                continue
    return out


def series_stats(series: List[float]) -> Dict[str, float]:
    if not series:
        return {"mean": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": sum(series) / len(series),
        "min": min(series),
        "max": max(series),
    }


def score_sample(sample: Dict[str, Any], pred_text: str) -> Dict[str, Any]:
    statistics = sample.get("statistics", {})
    pred = parse_prediction(pred_text)
    metrics: Dict[str, Any] = {"ok": True}

    for k in ("mean", "min", "max"):
        if k in statistics and k in pred:
            metrics[f"{k}_abs_err"] = abs(pred[k] - statistics[k])
        else:
            metrics["ok"] = False
    return metrics


def aggregate(scores: List[Dict[str, Any]]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    if not scores:
        return agg
    keys = [k for k in scores[0].keys() if k.endswith("_abs_err")]
    for k in keys:
        vals = [s.get(k) for s in scores if isinstance(s.get(k), (int, float))]
        if vals:
            agg[f"{k}_mean"] = sum(vals) / len(vals)
    agg["samples"] = float(len(scores))
    agg["ok_rate"] = sum(1 for s in scores if s.get("ok")) / max(1, len(scores))
    
    # Performance metric: average of mean, min, and max errors (lower is better)
    mean_err = agg.get("mean_abs_err_mean", 0.0)
    min_err = agg.get("min_abs_err_mean", 0.0)
    max_err = agg.get("max_abs_err_mean", 0.0)
    agg["performance"] = (mean_err + min_err + max_err) / 3.0
    
    return agg
