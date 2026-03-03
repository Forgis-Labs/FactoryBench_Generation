"""
Shared template-filling utilities for question generators across all levels.

Covers: safe string substitution, event description filling, signal picking,
chunk encoding/sampling, and context building.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.questions.common.time_series import (
    encode_time_series,
    format_note_value,
    remove_constant_features,
    remove_feature,
    sort_feature_keys,
    strip_null_features,
)


# ---------------------------------------------------------------------------
# Safe string substitution
# ---------------------------------------------------------------------------


def fill(template_str: str, **kwargs: Any) -> str:
    """
    Safe template substitution via str.replace().
    Avoids str.format() pitfalls with literal braces (e.g. {T}, {T+n}).
    """
    result = template_str
    for key, value in kwargs.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


# ---------------------------------------------------------------------------
# Timestamp and signal helpers
# ---------------------------------------------------------------------------


def get_last_timestamp(rows: List[Dict[str, Any]]) -> int:
    """Return the last timestamp_ms value in the rows as an integer."""
    for row in reversed(rows):
        ts = row.get("timestamp_ms")
        if ts is not None:
            try:
                return int(float(ts))
            except (TypeError, ValueError):
                pass
    return 0


def get_numeric_signal_names(rows: List[Dict[str, Any]]) -> List[str]:
    """Return signal names that hold numeric values, excluding metadata columns."""
    exclude = {"timestamp_ms", "fault_label"}
    if not rows:
        return []
    return [
        key
        for key in sorted(rows[0].keys())
        if key not in exclude and isinstance(rows[0].get(key), (int, float, np.floating))
    ]


def pick_scalar_signal(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Pick a random numeric signal name from the first row."""
    names = get_numeric_signal_names(rows)
    return random.choice(names) if names else None


def pick_joint_velocity_and_torque(
    rows: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float]]:
    """Return a representative velocity and torque value from the last row."""
    if not rows:
        return None, None
    last_row = rows[-1]
    velocity = None
    for key in sorted(last_row.keys()):
        if "velocity" in key or "speed" in key:
            val = last_row.get(key)
            if isinstance(val, (int, float, np.floating)):
                velocity = round(float(val), 3)
                break
    torque = None
    for key in sorted(last_row.keys()):
        if "torque" in key or "current" in key:
            val = last_row.get(key)
            if isinstance(val, (int, float, np.floating)):
                torque = round(float(val), 3)
                break
    return velocity, torque


# ---------------------------------------------------------------------------
# Chunk sampling and encoding
# ---------------------------------------------------------------------------


def encode_chunk(rows: List[Dict[str, Any]]) -> str:
    """Encode a list of rows as a single compact string for display in options."""
    stripped = strip_null_features(rows)
    stripped = remove_feature(stripped, "fault_label")
    stripped = sort_feature_keys(stripped)
    if not stripped:
        return "[]"
    encoded, _ = encode_time_series(stripped)
    return " | ".join(encoded)


def sample_chunks(
    rows: List[Dict[str, Any]],
    n_chunks: int = 4,
    min_chunk: int = 5,
    max_chunk: int = 7,
) -> List[List[Dict[str, Any]]]:
    """
    Sample n_chunks non-overlapping contiguous windows of random length
    [min_chunk, max_chunk] from rows.  Returns fewer than n_chunks if the
    rows are not long enough.
    """
    available = len(rows)
    if available < n_chunks * min_chunk:
        return []
    occupied: set = set()
    chunks: List[List[Dict[str, Any]]] = []
    max_attempts = n_chunks * 50
    attempts = 0
    while len(chunks) < n_chunks and attempts < max_attempts:
        attempts += 1
        chunk_len = random.randint(min_chunk, min(max_chunk, available))
        if available < chunk_len:
            continue
        start = random.randint(0, available - chunk_len)
        span = set(range(start, start + chunk_len))
        if not span & occupied:
            chunks.append(rows[start : start + chunk_len])
            occupied |= span
    return chunks


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------


def build_context(subseries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build the context dict attached to every generated question.
    Contains the encoded time series and any constant-feature notes.
    """
    ts = strip_null_features(subseries)
    ts = sort_feature_keys(ts)
    ts = remove_feature(ts, "fault_label")
    ts, constant_features = remove_constant_features(ts)
    ts = sort_feature_keys(ts)
    encoded, acronym_mapping = encode_time_series(ts)

    ctx: Dict[str, Any] = {}
    if constant_features:
        ctx["notes"] = {
            "disclaimer": "these features stayed constant at the following values",
            "constant_features": {
                k: format_note_value(constant_features[k])
                for k in sorted(constant_features.keys())
            },
        }
    ctx["time_series_format"] = {
        "description": "Each timestep is encoded as a string with features in format 'acronym_value' separated by '|'.",
        "acronym_mapping": acronym_mapping,
    }
    ctx["time_series"] = encoded
    return ctx


# ---------------------------------------------------------------------------
# Event description filling
# ---------------------------------------------------------------------------


def fill_event_description(
    event: Dict[str, Any],
    subseries: List[Dict[str, Any]],
    t: int,
) -> str:
    """
    Fill an event's description template with values derived from the subseries.

    Supported variable types (from events.json):
      feature_i  → a random numeric signal name
      X          → signal value at the start of the subseries
      Y          → signal value at the end of the subseries
      delta      → |end - start|
      duration   → random integer 2–10
      rate       → (end - start) / len(subseries)
      x          → random payload weight
      T          → the last timestamp of the subseries
    """
    desc: str = event["description"]
    variables: Dict[str, str] = event.get("variables", {})

    signal = pick_scalar_signal(subseries) or "joint_velocity_0"

    start_val: Optional[float] = None
    end_val: Optional[float] = None
    for row in subseries:
        v = row.get(signal)
        if isinstance(v, (int, float, np.floating)):
            start_val = round(float(v), 3)
            break
    for row in reversed(subseries):
        v = row.get(signal)
        if isinstance(v, (int, float, np.floating)):
            end_val = round(float(v), 3)
            break

    kwargs: Dict[str, Any] = {"T": t}

    if "feature_i" in variables:
        kwargs["feature_i"] = signal
    if "X" in variables:
        kwargs["X"] = start_val if start_val is not None else 0.0
    if "Y" in variables:
        kwargs["Y"] = end_val if end_val is not None else 0.0
    if "delta" in variables:
        kwargs["delta"] = round(abs((end_val or 0.0) - (start_val or 0.0)), 3)
    if "duration" in variables:
        kwargs["duration"] = random.randint(2, 10)
    if "rate" in variables:
        n = max(1, len(subseries))
        kwargs["rate"] = round(((end_val or 0.0) - (start_val or 0.0)) / n, 4)
    if "x" in variables:
        kwargs["x"] = random.choice([0.5, 1.0, 1.5, 2.0, 2.5])

    return fill(desc, **kwargs)


# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------


def discover_episodes_by_dataset(
    datasets_dir: Path,
    datasets: List[str],
) -> Dict[str, List[Path]]:
    """
    Return {dataset_name: [episode_path, ...]} for each dataset that has
    normalized episode JSON files under datasets_dir/normalized_episodes/<name>/.
    """
    by_dataset: Dict[str, List[Path]] = {}
    for ds in datasets:
        ep_dir = datasets_dir / "normalized_episodes" / ds
        if not ep_dir.exists():
            continue
        paths = [
            p
            for p in sorted(ep_dir.glob("*.json"))
            if not p.stem.endswith("_metadata")
        ]
        if paths:
            by_dataset[ds] = paths
    return by_dataset
