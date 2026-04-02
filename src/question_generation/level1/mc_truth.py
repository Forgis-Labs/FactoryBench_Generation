from __future__ import annotations

import math
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from huggingface_hub import hf_hub_download


def interpolate_value(t_target: float, t1: float, v1: float, t2: float, v2: float) -> float:
    """Linear interpolation between two values."""
    if t2 == t1:
        return v1
    alpha = (t_target - t1) / (t2 - t1)
    return (1 - alpha) * v1 + alpha * v2


_MACHINE_METADATA_CACHE: List[Dict[str, Any]] = None


def load_machine_metadata() -> List[Dict[str, Any]]:
    """Load machines.json from the data directory or Hugging Face."""
    global _MACHINE_METADATA_CACHE
    if _MACHINE_METADATA_CACHE is not None:
        return _MACHINE_METADATA_CACHE

    machines_path = Path("data/labelling/machines.json")
    try:
        if machines_path.exists():
            with open(machines_path, "r", encoding="utf-8") as f:
                _MACHINE_METADATA_CACHE = json.load(f)
        else:
            local_path = hf_hub_download(
                repo_id="Forgis/FactoryNet_Dataset", 
                repo_type="dataset", 
                filename="data/labelling/machines.json"
            )
            with open(local_path, "r", encoding="utf-8") as f:
                _MACHINE_METADATA_CACHE = json.load(f)
    except Exception:
        _MACHINE_METADATA_CACHE = []
    return _MACHINE_METADATA_CACHE


def get_machine_by_id(machine_id: int) -> Optional[Dict[str, Any]]:
    metadata = load_machine_metadata()
    for m in metadata:
        if m.get("machine_id") == machine_id:
            return m
    return None


def find_bracketing_indices(rows: List[Dict[str, Any]], t_ms: float) -> Tuple[Optional[int], Optional[int]]:
    """Find indices of samples bracketing time t_ms. Returns (before_idx, after_idx)."""
    timestamps = [(i, r.get("timestamp_ms")) for i, r in enumerate(rows) if r.get("timestamp_ms") is not None]
    
    if not timestamps:
        return None, None
    
    # Find exact match
    for i, ts in timestamps:
        if ts == t_ms:
            return i, i
    
    # Find bracketing
    before_idx, after_idx = None, None
    for i, ts in timestamps:
        if ts <= t_ms:
            before_idx = i
        if ts >= t_ms and after_idx is None:
            after_idx = i
            break
            
    # if it's past the end, snap to last
    if before_idx is not None and after_idx is None:
        after_idx = before_idx
    # if it's before the start, snap to first
    if before_idx is None and after_idx is not None:
        before_idx = after_idx
        
    return before_idx, after_idx


def interpolate_signal_at_time(
    rows: List[Dict[str, Any]],
    key: str,
    t_ms: float,
) -> Tuple[Optional[float], Optional[str]]:
    """Interpolate a signal at a given time. Returns (value, mode) or (None, None)."""
    before_idx, after_idx = find_bracketing_indices(rows, t_ms)
    if before_idx is None or after_idx is None:
        return None, None
    try:
        if before_idx == after_idx:
            return float(rows[before_idx][key]), "exact"
        v1 = float(rows[before_idx][key])
        v2 = float(rows[after_idx][key])
        t1 = float(rows[before_idx]["timestamp_ms"])
        t2 = float(rows[after_idx]["timestamp_ms"])
        return interpolate_value(t_ms, t1, v1, t2, v2), "interpolated"
    except (KeyError, TypeError, ValueError):
        return None, None


def get_wrench_components_at_time(
    rows: List[Dict[str, Any]],
    t_ms: float,
    indices: List[int],
) -> Tuple[Optional[List[float]], Optional[str], Optional[str]]:
    """Get wrench components (force/torque) at time, preferring external sensor then estimate."""
    # Try external sensor first
    values = []
    modes = []
    for idx in indices:
        val, mode = interpolate_signal_at_time(rows, f"true_force_{idx}", t_ms)
        if val is None:
            values = []
            break
        values.append(val)
        modes.append(mode)
    if values:
        interp_mode = "exact" if all(m == "exact" for m in modes) else "interpolated"
        return values, "external sensor", interp_mode

    # Fallback to controller estimate
    values = []
    modes = []
    for idx in indices:
        val, mode = interpolate_signal_at_time(rows, f"est_contact_force_{idx}", t_ms)
        if val is None:
            return None, None, None
        values.append(val)
        modes.append(mode)
    interp_mode = "exact" if all(m == "exact" for m in modes) else "interpolated"
    return values, "controller estimate", interp_mode


def get_num_joints(rows: List[Dict[str, Any]], signal_prefix: str = "feedback_pos_") -> int:
    """Detect the number of joints by inspecting columns in the first row."""
    if not rows:
        return 0
    first_row = rows[0]
    
    # Handle aliases
    prefixes = [signal_prefix]
    if signal_prefix == "feedback_speed_":
        prefixes.append("feedback_vel_")
    
    for pref in prefixes:
        count = 0
        while f"{pref}{count}" in first_row:
            count += 1
        if count > 0:
            return count
    return 0


def get_signal_key(row: Dict[str, Any], prefix: str, axis: int) -> str:
    """Get the key for a signal, handling aliases."""
    key = f"{prefix}{axis}"
    if key in row:
        return key
    
    # Aliases
    if prefix == "feedback_speed_":
        alt_key = f"feedback_vel_{axis}"
        if alt_key in row:
            return alt_key
    elif prefix == "vibration_":
        alt_key = f"auxiliary_accel_tool_{axis}"
        if alt_key in row:
            return alt_key
            
    return key


def _mode_lookup(machine: Dict[str, Any], mode_key: str, value: float) -> Optional[str]:
    """Look up a human-readable mode name from the machine KG enum."""
    mode_spec = machine.get(mode_key, {})
    enum_list = mode_spec.get("enum", [])
    int_val = int(round(value))
    for entry in enum_list:
        if entry.get("value") == int_val:
            return entry["name"]
    return None


def _mode_distractors(machine: Dict[str, Any], mode_key: str, correct_name: str, n: int = 3) -> List[str]:
    """Pick n random distractor mode names from the KG enum, excluding the correct one."""
    mode_spec = machine.get(mode_key, {})
    enum_list = mode_spec.get("enum", [])
    candidates = [e["name"] for e in enum_list if e["name"] != correct_name]
    return random.sample(candidates, min(n, len(candidates)))



def answer_q1_state_joint_moved(
    rows: List[Dict[str, Any]],
    t1_ms: float,
    t2_ms: float,
    axis: int,
    eps_1: float
) -> Dict[str, Any]:
    # A - Stationary (diff <= eps_1)
    # B - Higher positive (diff > eps_1)
    # C - Lower negative (diff < -eps_1)
    # D - Invalid

    pos_key = f"feedback_pos_{axis}"
    val_t1, _ = interpolate_signal_at_time(rows, pos_key, t1_ms)
    val_t2, _ = interpolate_signal_at_time(rows, pos_key, t2_ms)
    
    if val_t1 is None or val_t2 is None:
        return {
            "answer": "D", 
            "reasoning": "Missing telemetry data at requested timestamps.", 
            "is_true": False
        }

    diff = val_t2 - val_t1
    abs_diff = abs(diff)

    if abs_diff <= eps_1:
        correct_letter = "A"
    elif diff > eps_1:
        correct_letter = "B"
    else: # diff < -eps_1
        correct_letter = "C"

    options_dict = {
        "A": "The joint remained stationary (difference <= {eps_1}).",
        "B": "The joint moved to a higher positive angular position.",
        "C": "The joint moved to a lower negative angular position.",
        "D": "The telemetry data at these timestamps is missing or invalid."
    }

    return {
        "answer": correct_letter, 
        "options": options_dict,
        "reasoning": f"val_t1={val_t1:.4f}, val_t2={val_t2:.4f}, diff={diff:.4f}. eps={eps_1}. Correct: {correct_letter}",
        "raw_value": diff,
        "is_true": True
    }


def answer_q2_state_friction_increase(
    rows: List[Dict[str, Any]],
    t1_ms: float,
    t2_ms: float,
    axis: int,
    eps_2: float,
    delta_1_ms: int = 500,
) -> Dict[str, Any]:
    speed_key = get_signal_key(rows[0], "feedback_speed_", axis)
    current_key = f"effort_current_{axis}"
    
    def compute_friction_proxy(w_start: float, w_end: float) -> Optional[float]:
        window_rows = [r for r in rows if r.get("timestamp_ms") is not None and w_start <= r["timestamp_ms"] <= w_end]
        if len(window_rows) < 3:
            return None
            
        ratios = []
        for r in window_rows:
            try:
                v = float(r[speed_key])
                i = float(r[current_key])
                # avoid noise
                if abs(v) > 1e-6:
                    ratios.append(abs(i) / abs(v))
            except (KeyError, TypeError, ValueError):
                continue
                
        if not ratios:
            return None
        return float(np.median(ratios))

    f1 = compute_friction_proxy(t1_ms - delta_1_ms, t1_ms + delta_1_ms)
    f2 = compute_friction_proxy(t2_ms - delta_1_ms, t2_ms + delta_1_ms)
    
    if f1 is None or f2 is None:
        return {
            "answer": "D", 
            "reasoning": "Insufficient telemetry data for friction proxy calculation.", 
            "is_true": False
        }

    # percentage change calculation
    perc_change = ((f2 - f1) / f1) * 100.0 if f1 != 0 else float('inf')
    
    # Clasificación según el esquema JSON:
    # A - Increase (> eps_2)
    # B - Decrease (< -eps_2)
    # C - Stable (between +/- eps_2)
    # D - Fluctuation (fallback)
    
    if perc_change > eps_2:
        correct_letter = "A"
    elif perc_change < -eps_2:
        correct_letter = "B"
    else:
        correct_letter = "C"

    options_dict = {
        "A": f"Friction proxy increased significantly (> {eps_2}%).",
        "B": f"Friction proxy decreased significantly (< -{eps_2}%).",
        "C": f"Friction proxy remained stable (within +/- {eps_2}%).",
        "D": "Friction proxy fluctuated without a clear directional trend."
    }

    return {
        "answer": correct_letter, 
        "options": options_dict,
        "reasoning": f"f1={f1:.4f}, f2={f2:.4f}, change={perc_change:.2f}%. Threshold={eps_2}%. Correct: {correct_letter}",
        "raw_value": perc_change,
        "f1": f1,
        "f2": f2,
        "is_true": True
    }


def answer_q3_state_acceleration(
    rows: List[Dict[str, Any]], 
    t_ms: float, 
    threshold: float = 1.0
) -> Dict[str, Any]:
    vib0, _ = interpolate_signal_at_time(rows, "vibration_0", t_ms)
    vib1, _ = interpolate_signal_at_time(rows, "vibration_1", t_ms)
    vib2, _ = interpolate_signal_at_time(rows, "vibration_2", t_ms)

    if vib0 is not None and vib1 is not None and vib2 is not None:
        G = 9.81
        # convert to m/s^2 and round for tensor format
        a_x, a_y, a_z = round(vib0 * G, 2), round(vib1 * G, 2), round(vib2 * G, 2)
        magnitude = math.sqrt(a_x**2 + a_y**2 + a_z**2)
        
        answer_tensor = f"{a_x}_{a_y}_{a_z}"
        
        return {
            "answer": answer_tensor,
            "reasoning": f"End-effector accelerometer (m/s^2). Magnitude: {magnitude:.4f}.",
            "raw_value": [a_x, a_y, a_z],
            "magnitude": magnitude,
            "is_true": magnitude > threshold,
            "anchor_timestamps": {t_ms},
            "source": "accelerometer",
            "acceptance_bounds": {"margin": [0.5, 0.5, 0.5]}
        }

    # acceleration (rad/s^2)
    dt_ms = 10.0
    accels: List[float] = []
    
    num_j = get_num_joints(rows, "feedback_speed_")
    for axis in range(min(3, num_j)):
        speed_key = get_signal_key(rows[0], "feedback_speed_", axis)
        v1, _ = interpolate_signal_at_time(rows, speed_key, t_ms)
        v2, _ = interpolate_signal_at_time(rows, speed_key, t_ms + dt_ms)
        
        if v1 is None or v2 is None:
            continue
            
        # a = (v2 - v1) / dt
        accel_val = (v2 - v1) / (dt_ms / 1000.0)
        accels.append(round(accel_val, 2))

    if len(accels) < 3:
        return {
            "answer": "N/A", 
            "reasoning": "Insufficient data to compute 3-axis acceleration.", 
            "is_true": False
        }

    a_0, a_1, a_2 = accels[0], accels[1], accels[2]
    magnitude = math.sqrt(a_0**2 + a_1**2 + a_2**2)
    answer_tensor = f"{a_0}_{a_1}_{a_2}"

    return {
        "answer": answer_tensor,
        "reasoning": f"Joint angular acceleration (rad/s^2) from speed diff. Magnitude: {magnitude:.4f}.",
        "raw_value": [a_0, a_1, a_2],
        "magnitude": magnitude,
        "is_true": magnitude > threshold,
        "anchor_timestamps": {t_ms, t_ms + dt_ms},
        "source": "finite_differences",
        "acceptance_bounds": {"margin": [0.5, 0.5, 0.5]}
    }


def answer_q4_state_external_force_detected(
    rows: List[Dict[str, Any]], 
    t_ms: float, 
    eps_3: float
) -> Dict[str, Any]:
    # A - No significative (<= eps_3)
    # B - Positive  (> eps_3 y dir > 0)
    # C - Negative (> eps_3 y dir < 0)
    # D - Sensor out of bounds/unavailable
    values, source, _ = get_wrench_components_at_time(rows, t_ms, [0, 1, 2])
    
    magnitude = 0.0
    dominant_direction = 0.0
    source_label = ""

    if values is not None:
        fx, fy, fz = values
        magnitude = math.sqrt(fx**2 + fy**2 + fz**2)
        # get dominant direction by largest absolute component
        components = [fx, fy, fz]
        dominant_direction = components[int(np.argmax([abs(x) for x in components]))]
        source_label = source
    else:
        # effort current as proxy (Fallback Path)
        currents = []
        num_j = get_num_joints(rows, "feedback_speed_")
        for idx in range(min(3, num_j)):
            val, _ = interpolate_signal_at_time(rows, f"effort_current_{idx}", t_ms)
            if val is not None:
                currents.append(val)
        
        if not currents:
            return {
                "answer": "D", 
                "reasoning": "Sensor data and current proxy are unavailable.", 
                "is_true": False
            }
            
        magnitude = math.sqrt(sum(c**2 for c in currents))
        dominant_direction = sum(currents)
        source_label = "current proxy"

    if magnitude <= eps_3:
        correct_letter = "A"
    elif dominant_direction >= 0:
        correct_letter = "B"
    else:
        correct_letter = "C"

    options_dict = {
        "A": f"No significant force/torque detected (magnitude <= {eps_3}).",
        "B": "Force/torque detected above threshold in the positive direction.",
        "C": "Force/torque detected above threshold in the negative direction.",
        "D": "Sensor data is out of bounds or unavailable."
    }

    return {
        "answer": correct_letter,
        "options": options_dict,
        "reasoning": f"Source: {source_label}. Magnitude {magnitude:.4f} (Threshold: {eps_3}). Dom. Direction: {dominant_direction:.4f}. Correct: {correct_letter}.",
        "raw_value": magnitude,
        "source": source_label,
        "is_true": magnitude > eps_3
    }


def answer_q5_state_signal_statistic(
    rows: List[Dict[str, Any]], 
    t1_ms: float, 
    t2_ms: float, 
    axis: int, 
    signal_name: str, 
    statistic: str = "mean"
) -> Dict[str, Any]:
    """Calculate a basic statistic (mean, max, min) over a time window."""
    prefix_map = {
        "effort_current": "effort_current_",
        "feedback_speed": "feedback_speed_",
        "feedback_pos": "feedback_pos_"
    }
    
    key = get_signal_key(rows[0], prefix_map.get(signal_name, "feedback_pos_"), axis)

    window_rows = [r for r in rows if r.get("timestamp_ms") is not None and t1_ms <= r["timestamp_ms"] <= t2_ms]
    
    vals = []
    for r in window_rows:
        try:
            if key in r and r[key] is not None:
                vals.append(float(r[key]))
        except (ValueError, TypeError):
            continue

    if not vals:
        return {
            "answer": "N/A", 
            "reasoning": f"No valid data for {key} in window [{t1_ms}, {t2_ms}].", 
            "is_true": False
        }

    if statistic == "mean":
        val = sum(vals) / len(vals)
    elif statistic == "max":
        val = max(vals)
    elif statistic == "min":
        val = min(vals)
    else:
        return {"answer": "Error", "reasoning": f"Unknown statistic type: {statistic}", "is_true": False}

    val_rounded = round(val, 4)

    return {
        "answer": str(val_rounded),
        "reasoning": f"Calculated {statistic} using {len(vals)} samples for {key}. Window: {t1_ms}-{t2_ms}ms.",
        "raw_value": val_rounded,
        "is_true": True,
        "anchor_timestamps": [t1_ms, t2_ms],
        "important_features": [key],
        "acceptance_bounds": {"margin": 0.05}
    }


def answer_q6_state_joint_speed_ranking(
    rows: List[Dict[str, Any]], 
    t_ms: float, 
    joints_to_rank: List[int]
) -> Dict[str, Any]:
    """Rank specific joints by absolute speed at time T."""
    signal_prefix = "feedback_speed_"
    
    if len(joints_to_rank) != 4:
        return {
            "answer": "N/A", 
            "reasoning": "Ranking requires exactly 4 joints.", 
            "is_true": False
        }
        
    values = []
    # A -> joints_to_rank[0], B -> joints_to_rank[1], C -> joints_to_rank[2], D -> joints_to_rank[3]
    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    
    for i, axis in enumerate(joints_to_rank):
        key = get_signal_key(rows[0], signal_prefix, axis)
        val, _ = interpolate_signal_at_time(rows, key, t_ms)
        
        if val is None:
            return {
                "answer": "Unknown", 
                "reasoning": f"Missing data for joint {axis} at {t_ms}ms.", 
                "is_true": False
            }
        
        values.append((label_map[i], abs(val), axis))
    
    values.sort(key=lambda x: x[1], reverse=True)

    ranking_str = "".join(item[0] for item in values)

    options_dict = {label_map[i]: f"Joint {axis}" for i, axis in enumerate(joints_to_rank)}

    return {
        "answer": ranking_str,
        "options": options_dict,
        "reasoning": f"Speeds: " + ", ".join([f"{item[0]}(J{item[2]}): {item[1]:.4f}" for item in values]),
        "is_true": True,
        "anchor_timestamps": [t_ms]
    }


def answer_q7_state_joint_within_rated_speed(rows: List[Dict[str, Any]], t_ms: float, machine_id: int, joints_list: List[int]) -> Dict[str, Any]:
    """Multi-select TFTF speed limit check."""
    machine = get_machine_by_id(machine_id)
    if not machine or "joint_speed_limits" not in machine:
        return {"answer": "Unknown", "reasoning": f"No speed limit metadata for machine {machine_id}", "is_true": False}

    limits = machine["joint_speed_limits"]
    results = []
    reasoning_parts = []
    
    all_true = True
    for axis in joints_list:
        speed_key = get_signal_key(rows[0], "feedback_speed_", axis)
        val, _ = interpolate_signal_at_time(rows, speed_key, t_ms)
        if val is None:
            return {"answer": "Unknown", "reasoning": f"Missing speed data for joint {axis}", "is_true": False}
        
        limit = limits[axis] if axis < len(limits) else 3.14
        is_within = abs(val) <= limit
        all_true = all_true and is_within
        results.append("T" if is_within else "F")
        reasoning_parts.append(f"J{axis}: |{val:.2f}| <= {limit}")

    answer = "".join(results)
    
    options_dict = {
        "A": f"Joint {joints_list[0]} is within its rated maximum speed.",
        "B": f"Joint {joints_list[1]} is within its rated maximum speed.",
        "C": f"Joint {joints_list[2]} is within its rated maximum speed.",
        "D": f"Joint {joints_list[3]} is within its rated maximum speed."
    }
    
    return {
        "answer": answer,
        "options": options_dict,
        "reasoning": "; ".join(reasoning_parts),
        "raw_results": results,
        "is_true": all_true
    }


def answer_q8_state_current_within_rated(rows: List[Dict[str, Any]], t_ms: float, axis: int, machine_id: int) -> Dict[str, Any]:
    """Map to A/B/C/D based on current limit check."""
    machine = get_machine_by_id(machine_id)
    if not machine or "rated_current_per_joint" not in machine:
        return {"answer": "Unknown", "reasoning": f"No current metadata for machine {machine_id}", "is_true": False}

    limits = machine["rated_current_per_joint"]
    limit = limits[axis] if axis < len(limits) else 2.0
    
    current_key = f"effort_current_{axis}"
    val, _ = interpolate_signal_at_time(rows, current_key, t_ms)
    
    options_dict = {
        "A": "Current is safely within nominal limits (<= 80% of rated).",
        "B": "Current is near the limit (80% - 100% of rated).",
        "C": "Current exceeds the continuous rated limit (> 100%).",
        "D": "Current reading is unexpectedly zero or missing."
    }

    if val is None or val == 0.0:
        return {
            "answer": "D", 
            "options": options_dict, 
            "reasoning": "Missing or zero current data", 
            "is_true": False
        }

    abs_val = abs(val)
    ratio = abs_val / limit

    if ratio <= 0.8:
        correct_letter = "A"
    elif ratio <= 1.0:
        correct_letter = "B"
    else:
        correct_letter = "C"

    return {
        "answer": correct_letter,
        "options": options_dict,
        "reasoning": f"Current |{val:.2f}A| (limit: {limit}A). Ratio: {ratio*100:.1f}%. Option {correct_letter}.",
        "is_true": True,
        "raw_value": val
    }


def answer_q9_state_signal_description(
    rows: List[Dict[str, Any]], 
    t1_ms: float, 
    t2_ms: float, 
    axis: int, 
    signal_name: str
) -> Dict[str, Any]:
    """Identify signal behaviour from standardized options A/B/C/D."""
    prefix_map = {
        "effort_current": "effort_current_",
        "feedback_speed": "feedback_speed_",
        "feedback_pos": "feedback_pos_"
    }
    key = get_signal_key(rows[0], prefix_map.get(signal_name, "feedback_pos_"), axis)

    window_rows = [r for r in rows if r.get("timestamp_ms") is not None and t1_ms <= r["timestamp_ms"] <= t2_ms]
    
    if len(window_rows) < 5:
        return {"answer": "D", "reasoning": "Insufficient samples to determine trend.", "is_true": False}

    vals = [float(r[key]) for r in window_rows if key in r]
    if not vals:
        return {"answer": "D", "reasoning": "No valid telemetry data in window.", "is_true": False}

    v_start, v_end = vals[0], vals[-1]
    v_min, v_max = min(vals), max(vals)
    delta = v_end - v_start
    range_val = v_max - v_min

    is_erratic = range_val > 3 * abs(delta) and range_val > 0.01

    if is_erratic:
        correct_letter = "D"
    elif abs(delta) < 0.01 * (abs(v_min) + 1e-6) or range_val < 1e-4:
        correct_letter = "A"
    elif delta > 0:
        correct_letter = "B"
    else:
        correct_letter = "C"

    options_dict = {
        "A": "The signal is generally stable (fluctuations within a normal noise threshold).",
        "B": "The signal exhibits a clear and continuous increasing trend.",
        "C": "The signal exhibits a clear and continuous decreasing trend.",
        "D": "The signal is highly erratic or oscillating without a single directional trend."
    }

    return {
        "answer": correct_letter,
        "options": options_dict,
        "reasoning": f"delta={delta:.4f}, range={range_val:.4f}, erratic={is_erratic}. Correct: {correct_letter}.",
        "raw_stats": {"delta": delta, "range": range_val, "samples": len(vals)},
        "is_true": True
    }


_OPERATIONAL_STATES = ["Idle", "Standby", "Normal Operation", "High Load"]


def _derive_operational_state(rows: List[Dict[str, Any]], t_ms: float) -> Optional[str]:
    """Derive the operational state from joint speed & current patterns."""
    num_j = get_num_joints(rows, "feedback_speed_")
    if num_j == 0:
        return None

    speeds: List[float] = []
    currents: List[float] = []
    for axis in range(min(num_j, 6)):
        speed_key = get_signal_key(rows[0], "feedback_speed_", axis)
        v, _ = interpolate_signal_at_time(rows, speed_key, t_ms)
        c, _ = interpolate_signal_at_time(rows, f"effort_current_{axis}", t_ms)
        if v is not None:
            speeds.append(abs(v))
        if c is not None:
            currents.append(abs(c))

    if not speeds or not currents:
        return None

    max_speed = max(speeds)
    max_current = max(currents)

    if max_speed < 0.01 and max_current < 0.05:
        return "Idle"
    if max_speed < 0.01 and max_current >= 0.05:
        return "Standby"
    if max_current < 2.0:
        return "Normal Operation"
    return "High Load"


def answer_q10_state_safety_mode(rows: List[Dict[str, Any]], t_ms: float, machine_id: int) -> Dict[str, Any]:
    """Identify the robot's operational / safety mode at time T."""
    machine = get_machine_by_id(machine_id)

    # Primary path: safety_mode column + KG enum
    val, _ = interpolate_signal_at_time(rows, "safety_mode", t_ms)
    if val is not None and machine and "safety_modes" in machine:
        correct_name = _mode_lookup(machine, "safety_modes", val)
        if correct_name is not None:
            distractors = _mode_distractors(machine, "safety_modes", correct_name, n=3)
            if len(distractors) < 3:
                distractors += [f"Mode_{i}" for i in range(3 - len(distractors))]
            option_list = [correct_name] + distractors
            random.shuffle(option_list)
            correct_letter = chr(ord("A") + option_list.index(correct_name))
            options = {chr(ord("A") + i): f"The robot is in {name}." for i, name in enumerate(option_list)}
            return {
                "answer": correct_letter,
                "options": options,
                "reasoning": f"safety_mode={int(round(val))} maps to '{correct_name}' in KG. Correct option is {correct_letter}.",
                "is_true": True,
            }

    # Fallback: derive operational state from speed & current patterns
    state = _derive_operational_state(rows, t_ms)
    if state is None:
        return {"answer": "Unknown", "reasoning": "Cannot derive operational state", "is_true": False}

    distractors = [s for s in _OPERATIONAL_STATES if s != state]
    random.shuffle(distractors)
    distractors = distractors[:3]

    option_list = [state] + distractors
    random.shuffle(option_list)
    correct_letter = chr(ord("A") + option_list.index(state))
    options = {chr(ord("A") + i): f"The robot is in {name}." for i, name in enumerate(option_list)}

    return {
        "answer": correct_letter,
        "options": options,
        "reasoning": f"Derived operational state: '{state}' from speed/current patterns. Correct option is {correct_letter}.",
        "is_true": True,
    }
