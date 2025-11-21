from typing import Any, Dict, List, Optional
import json
import os
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # type: ignore


def load_telemetry_literacy(
    source: str = "local",
    path: str = "datasets/basic_statistics.json",
    hf_slug: Optional[str] = None,
    split: str = "train",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if source == "hf":
        if load_dataset is None:
            raise RuntimeError("datasets library not installed; run: pip install datasets")
        if not hf_slug:
            raise ValueError("hf_slug required for source='hf'")
        try:
            # Use HF_API_TOKEN if available for private/gated datasets
            token = os.getenv("HF_API_TOKEN")
            # Don't pass empty token (causes auth errors)
            token = token if token and token.strip() else None
            # Load dataset without streaming (streaming returns malformed metadata)
            ds = load_dataset(hf_slug, split=split, streaming=False, token=token)
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                raise RuntimeError(f"Authentication failed for '{hf_slug}'. Check HF_API_TOKEN or dataset visibility.")
            elif "404" in error_msg or "not found" in error_msg.lower():
                raise RuntimeError(f"Dataset '{hf_slug}' not found on HuggingFace Hub.")
            else:
                raise RuntimeError(f"Failed to load HuggingFace dataset '{hf_slug}': {type(e).__name__}: {e}")
        
        rows: List[Dict[str, Any]] = []
        # Determine how many samples to process
        num_samples = limit if limit else len(ds)
        num_samples = min(num_samples, len(ds))  # Don't exceed dataset size
        
        for idx in range(num_samples):
            r = ds[idx]
            
            try:
                # Extract statistics dict
                stats = r.get("statistics", {})
                stats_dict = {
                    "mean": float(stats.get("mean", 0.0)),
                    "std": float(stats.get("std", 0.0)),
                    "min": float(stats.get("min", 0.0)),
                    "max": float(stats.get("max", 0.0)),
                }
                
                row = {
                    "id": r.get("id"),
                    "timestamps": list(r.get("timestamps", [])),
                    "values": list(r.get("values", [])),
                    "domain": r.get("domain", "unknown"),
                    "subtype": r.get("subtype", "unknown"),
                    "statistics": stats_dict,
                }
                rows.append(row)
            except Exception as e:
                print(f"Warning: Skipping malformed sample {idx}: {e}")
                continue
        
        if len(rows) == 0:
            raise RuntimeError(f"No valid samples loaded from HuggingFace dataset '{hf_slug}'")
        
        return rows

    # Local JSON file
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local dataset not found: {p}")
    
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data[:limit] if limit else data
