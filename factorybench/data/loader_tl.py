from typing import Any, Dict, List, Optional
import json
import os
from pathlib import Path

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
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
        for r in ds:
            # HF dataset structure: id, timestamps, values, domain, subtype, statistics
            rows.append(
                {
                    "id": r.get("id"),
                    "timestamps": list(r.get("timestamps", [])),
                    "values": list(r.get("values", [])),
                    "domain": r.get("domain"),
                    "subtype": r.get("subtype"),
                    "statistics": dict(r.get("statistics", {})),
                }
            )
            if limit and len(rows) >= limit:
                break
        return rows

    # local - new format only (timestamps, values, statistics)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data[:limit] if limit else data
