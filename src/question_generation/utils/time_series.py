"""
Shared time-series processing utilities for question generators across all levels.

Covers: subseries sampling, feature encoding, constant-feature removal,
inactivity detection, and provenance helpers.
"""
from __future__ import annotations

import json
import math
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

INACTIVE_CONSTANT_THRESHOLD = 55
INACTIVITY_TRIM = 5


def parse_event_id(value: Any) -> int:
    """
    Parse an event identifier from either:
    - integer-like value (e.g., 0, 3, "4")
    - formatted string "i_v1_v2_..." where i is the event id
    """
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    s = str(value).strip()
    if not s:
        return 0
    head = s.split("_", 1)[0]
    try:
        return int(head)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Basic row transformations
# ---------------------------------------------------------------------------


def strip_null_features(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in row.items() if v is not None} for row in rows]


def sort_feature_keys(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: row[k] for k in sorted(row.keys())} for row in rows]


def remove_feature(rows: List[Dict[str, Any]], feature: str) -> List[Dict[str, Any]]:
    return [{k: v for k, v in row.items() if k != feature} for row in rows]


def format_note_value(value: Any) -> Any:
    if isinstance(value, (int, float, np.floating)):
        rounded = round(float(value), 2)
        if rounded.is_integer():
            return int(rounded)
        return rounded
    return value


# ---------------------------------------------------------------------------
# Constant-feature removal
# ---------------------------------------------------------------------------


def remove_constant_features(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Separate constant features from varying ones.
    timestamp_ms is never treated as constant.
    Returns (filtered_rows, constant_feature_dict).
    """
    if not rows:
        return rows, {}
    constants: Dict[str, Any] = {}
    keys = set(rows[0].keys())
    for key in keys:
        if key == "timestamp_ms":
            continue
        values: List[Any] = [row.get(key) for row in rows if row.get(key) is not None]
        if not values:
            continue
        if all(isinstance(v, (int, float, np.floating)) for v in values):
            series = np.array([float(v) for v in values], dtype=float)
            if series.size < 2:
                constants[key] = float(series.mean())
                continue
            min_val, max_val = float(series.min()), float(series.max())
            mean_val = float(series.mean())
            if math.isclose(max_val, min_val):
                constants[key] = mean_val
                continue
            std_val = float(series.std())
            eps = 1e-9
            rel_range = (max_val - min_val) / (abs(mean_val) + eps)
            cv = std_val / (abs(mean_val) + eps)
            if rel_range < 0.02 or cv < 0.01:
                constants[key] = mean_val
            continue
        first_value = values[0]
        if all(v == first_value for v in values[1:]):
            constants[key] = first_value
    if not constants:
        return rows, {}
    filtered = [{k: v for k, v in row.items() if k not in constants} for row in rows]
    return filtered, constants


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _create_feature_acronyms(feature_names: List[str]) -> Dict[str, str]:
    def make_acronym(name: str, expansion: int = 0) -> str:
        match = re.match(r"^(.+?)_(\d+)$", name)
        if match:
            base_name = match.group(1)
            number = match.group(2)
            words = base_name.split("_")
        else:
            words = name.split("_")
            number = ""
        acro = "".join(word[0].lower() for word in words if word)
        last_word = words[-1] if words else ""
        if expansion > 0 and len(last_word) > 1:
            for i in range(1, min(1 + expansion, len(last_word))):
                acro += last_word[i].lower()
        return f"{acro}{number}" if number else acro

    expansion = 0
    acronyms: Dict[str, str] = {}
    while True:
        acronyms = {name: make_acronym(name, expansion) for name in feature_names}
        if len(set(acronyms.values())) == len(feature_names):
            break
        expansion += 1
        if expansion > 10:
            break
    return acronyms


def _encode_timestep(row: Dict[str, Any], acronyms: Dict[str, str]) -> str:
    timestamp_part = ""
    parts = []
    if "timestamp_ms" in row:
        value = row["timestamp_ms"]
        if value is not None and isinstance(value, (int, float, np.floating)):
            rounded = round(float(value), 2)
            if rounded.is_integer():
                rounded = int(rounded)
            timestamp_part = f"t={rounded}"
    for feature_name in sorted(row.keys()):
        if feature_name == "timestamp_ms":
            continue
        value = row[feature_name]
        if value is None:
            continue
        acro = acronyms.get(feature_name, feature_name)
        if isinstance(value, (int, float, np.floating)):
            rounded = round(float(value), 2)
            if rounded.is_integer():
                rounded = int(rounded)
            parts.append(f"{acro}={rounded}")
        elif isinstance(value, str):
            parts.append(f"{acro}={value}")
        elif isinstance(value, dict):
            parts.append(f"{acro}={json.dumps(value)}")

    if timestamp_part and parts:
        return f"{timestamp_part}: " + ", ".join(parts)
    if timestamp_part:
        return timestamp_part
    return ", ".join(parts)


def encode_time_series(
    rows: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, str]]:
    """
    Encode a list of row dicts as compact strings.
    Returns (encoded_rows, reverse_acronym_mapping).
    """
    if not rows:
        return [], {}
    feature_names = sorted(rows[0].keys())
    acronyms = _create_feature_acronyms(feature_names)
    encoded = [_encode_timestep(row, acronyms) for row in rows]
    reverse_mapping = {v: k for k, v in acronyms.items()}
    return encoded, reverse_mapping


# ---------------------------------------------------------------------------
# Subseries sampling
# ---------------------------------------------------------------------------


def _normalize_timestamps(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Shift timestamps so the first row starts at 0."""
    if not rows or "timestamp_ms" not in rows[0]:
        return rows
    first_ts = rows[0].get("timestamp_ms")
    if first_ts is None:
        return rows
    try:
        first_ts = float(first_ts)
        return [
            {
                **row,
                "timestamp_ms": float(row.get("timestamp_ms", 0)) - first_ts
                if row.get("timestamp_ms") is not None
                else None,
            }
            for row in rows
        ]
    except (TypeError, ValueError):
        return rows


def sample_subseries(
    rows: List[Dict[str, Any]], min_len: int, max_len: int
) -> List[Dict[str, Any]]:
    """Sample a random contiguous subseries and normalize its timestamps to start at 0."""
    if not rows:
        return []
    length = len(rows)
    size = random.randint(min_len, min(max_len, length))
    if size <= 0:
        return rows
    start = random.randint(0, length - size)
    return _normalize_timestamps(rows[start : start + size])


def sample_subseries_with_remainder(
    rows: List[Dict[str, Any]], min_len: int, max_len: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Sample a random subseries and return the rows that follow it.
    Returns (subseries, post_event_rows).
    """
    if not rows:
        return [], []
    length = len(rows)
    size = random.randint(min_len, min(max_len, length))
    start = random.randint(0, length - size)
    return _normalize_timestamps(rows[start : start + size]), rows[start + size :]


def find_event_starts(rows: List[Dict[str, Any]]) -> List[int]:
    """
    Return indices where the 'event' field transitions from 0 (or absent) to non-zero.
    """
    starts = []
    for i, row in enumerate(rows):
        ev = parse_event_id(row.get("event", 0))
        prev_ev = parse_event_id(rows[i - 1].get("event", 0)) if i > 0 else 0
        if ev != 0 and prev_ev == 0:
            starts.append(i)
    return starts


def sample_subseries_before_event(
    rows: List[Dict[str, Any]],
    min_len: int,
    max_len: int,
    min_post_event_after: int = 0,
    return_metadata: bool = False,
) -> Union[
    Tuple[List[Dict[str, Any]], List[Dict[str, Any]]],
    Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int],
]:
    """
    Pick a random event start (where 'event' goes from 0 to non-zero), then
    sample a subseries of length in [min_len, max_len] ending just before it.

    Returns (subseries, post_event_rows) where post_event_rows starts at the
    event onset.  Returns ([], []) if no valid event exists or if the chosen
    event does not have enough preceding rows (caller should retry).
    """
    starts = find_event_starts(rows)
    if not starts:
        if return_metadata:
            return [], [], -1, 0
        return [], []

    valid_starts = [
        idx
        for idx in starts
        if idx >= min_len and (len(rows) - idx - 1) >= max(0, int(min_post_event_after))
    ]
    if not valid_starts:
        if return_metadata:
            return [], [], -1, 0
        return [], []

    event_start = random.choice(valid_starts)
    if event_start < min_len:
        if return_metadata:
            return [], [], -1, 0
        return [], []

    length = random.randint(min_len, min(max_len, event_start))
    start_idx = event_start - length
    subseries = _normalize_timestamps(rows[start_idx:event_start])
    post_event_rows = rows[event_start:]
    if return_metadata:
        return subseries, post_event_rows, start_idx, length
    return (
        subseries,
        post_event_rows,
    )


# ---------------------------------------------------------------------------
# Inactivity detection
# ---------------------------------------------------------------------------


def is_inactive_subseries(
    rows: List[Dict[str, Any]],
    threshold: int = INACTIVE_CONSTANT_THRESHOLD,
) -> bool:
    if not rows:
        return True
    trimmed = (
        rows[INACTIVITY_TRIM:-INACTIVITY_TRIM]
        if len(rows) > INACTIVITY_TRIM * 2
        else rows
    )
    ts = strip_null_features(trimmed)
    ts = remove_feature(ts, "fault_label")
    ts = remove_feature(ts, "event")
    _, constants = remove_constant_features(ts)
    return len(constants) >= threshold


# ---------------------------------------------------------------------------
# Fault label
# ---------------------------------------------------------------------------


def pick_fault_label(rows: List[Dict[str, Any]]) -> int:
    labels: List[int] = []
    for row in rows:
        val = row.get("fault_label")
        if val is None:
            continue
        try:
            labels.append(int(val))
        except (TypeError, ValueError):
            continue
    if not labels:
        return 0
    return max(set(labels), key=labels.count)
