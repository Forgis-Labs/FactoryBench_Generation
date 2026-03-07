from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


def interpolate_value(t_target: float, t1: float, v1: float, t2: float, v2: float) -> float:
    """Linear interpolation between two values."""
    if t2 == t1:
        return v1
    alpha = (t_target - t1) / (t2 - t1)
    return (1 - alpha) * v1 + alpha * v2


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


def answer_q1_position_check(
    rows: List[Dict[str, Any]],
    t1_ms: float,
    t2_ms: float,
    axis: int,
    eps_1: float
) -> Dict[str, Any]:
    pos_key = f"feedback_pos_{axis}"
    val_t1, mode_t1 = interpolate_signal_at_time(rows, pos_key, t1_ms)
    val_t2, mode_t2 = interpolate_signal_at_time(rows, pos_key, t2_ms)
    
    if val_t1 is None or val_t2 is None:
        return {"answer": "Unknown", "reasoning": "Missing signal values"}

    delta_q = abs(val_t2 - val_t1)
    moved = delta_q > eps_1
    return {"answer": "No" if moved else "Yes", "reasoning": f"Δq={delta_q:.6f} > {eps_1} -> {moved}"}


def answer_q2_friction_increase(
    rows: List[Dict[str, Any]],
    t1_ms: float,
    t2_ms: float,
    axis: int,
    eps_2: float,
    delta_1_ms: int = 500,
) -> Dict[str, Any]:
    speed_key = f"feedback_speed_{axis}"
    current_key = f"effort_current_{axis}"
    
    timestamps = [r.get("timestamp_ms") for r in rows if r.get("timestamp_ms") is not None]
    if not timestamps:
        return {"answer": "Unknown", "reasoning": "No valid timestamps"}
        
    def compute_friction_proxy(w_start: float, w_end: float) -> Optional[float]:
        window_rows = [r for r in rows if r.get("timestamp_ms") is not None and w_start <= r["timestamp_ms"] <= w_end]
        if len(window_rows) < 3:
            return None
            
        ratios = []
        for r in window_rows:
            try:
                v = float(r[speed_key])
                i = float(r[current_key])
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
        return {"answer": "Unknown", "reasoning": "Missing samples for proxy calculation"}

    ratio = f2 / f1 if f1 != 0 else float('inf')
    threshold_ratio = 1.0 + (eps_2 / 100.0)
    increased = ratio > threshold_ratio
    return {"answer": "Yes" if increased else "No", "reasoning": f"f2/f1={ratio:.4f} > {threshold_ratio:.4f}"}


def answer_q3_end_effector_accel(rows: List[Dict[str, Any]], t_ms: float) -> Dict[str, Any]:
    vib0, _ = interpolate_signal_at_time(rows, "vibration_0", t_ms)
    vib1, _ = interpolate_signal_at_time(rows, "vibration_1", t_ms)
    vib2, _ = interpolate_signal_at_time(rows, "vibration_2", t_ms)
    
    if vib0 is None or vib1 is None or vib2 is None:
        return {"answer": "Unknown", "reasoning": "Missing vibration data"}
        
    G = 9.81
    return {
        "answer": f"{vib0*G:.4f}_{vib1*G:.4f}_{vib2*G:.4f}",
        "reasoning": "Calculated from vibration sensors."
    }


def answer_q4_external_force(rows: List[Dict[str, Any]], t_ms: float, eps_3: float) -> Dict[str, Any]:
    values, source, _ = get_wrench_components_at_time(rows, t_ms, [0, 1, 2])
    if values is None:
        return {"answer": "Unknown", "reasoning": "Missing external force data"}

    fx, fy, fz = values
    magnitude = math.sqrt(fx**2 + fy**2 + fz**2)
    detected = magnitude >= eps_3
    return {"answer": "Yes" if detected else "No", "reasoning": f"Magnitude {magnitude:.4f} >= {eps_3}"}


def answer_q5_joint_jerk(rows: List[Dict[str, Any]], t_ms: float, axis: int) -> Dict[str, Any]:
    valid = [(i, r.get("timestamp_ms")) for i, r in enumerate(rows) if r.get("timestamp_ms") is not None]
    if len(valid) < 5:
        return {"answer": "Unknown", "reasoning": "Insufficient elements"}

    closest_pos = min(range(len(valid)), key=lambda p: abs(valid[p][1] - t_ms))
    if closest_pos < 2 or closest_pos + 2 >= len(valid):
        return {"answer": "Unknown", "reasoning": "Boundary limits"}

    idx_km2, t_km2 = valid[closest_pos - 2]
    idx_km1, t_km1 = valid[closest_pos - 1]
    idx_k, t_k = valid[closest_pos]
    idx_kp1, t_kp1 = valid[closest_pos + 1]
    idx_kp2, t_kp2 = valid[closest_pos + 2]

    speed_key = f"feedback_speed_{axis}"
    try:
        v_km2 = float(rows[idx_km2][speed_key])
        v_k = float(rows[idx_k][speed_key])
        v_kp2 = float(rows[idx_kp2][speed_key])

        dt_k_km2 = (t_k - t_km2) / 1000.0
        dt_kp2_k = (t_kp2 - t_k) / 1000.0
        if dt_k_km2 <= 0 or dt_kp2_k <= 0:
            return {"answer": "Unknown", "reasoning": "Invalid timestamps"}

        accel_km1 = (v_k - v_km2) / dt_k_km2
        accel_kp1 = (v_kp2 - v_k) / dt_kp2_k

        dt_kp1_km1 = (t_kp1 - t_km1) / 1000.0
        if dt_kp1_km1 <= 0:
            return {"answer": "Unknown", "reasoning": "Invalid timestamp"}

        jerk = (accel_kp1 - accel_km1) / dt_kp1_km1
    except (KeyError, TypeError, ValueError):
        return {"answer": "Unknown", "reasoning": "Missing features"}

    return {"answer": str(round(jerk, 4)), "reasoning": "Calculated central difference"}


def answer_q6_torque_magnitude(rows: List[Dict[str, Any]], t_ms: float, axis_label: str) -> Dict[str, Any]:
    axis_map = {"x": 3, "y": 4, "z": 5}
    if axis_label not in axis_map:
        return {"answer": "Unknown", "reasoning": "Invalid axis"}

    idx = axis_map[axis_label]
    values, _, _ = get_wrench_components_at_time(rows, t_ms, [idx])
    if values is None:
        return {"answer": "Unknown", "reasoning": "Missing data"}

    return {"answer": str(round(abs(values[0]), 4)), "reasoning": "Calculated absolute torque"}


def answer_q7_joint_speed_ranking(rows: List[Dict[str, Any]], t_ms: float, joints_list: List[int]) -> Dict[str, Any]:
    speeds = []
    before_idx, after_idx = find_bracketing_indices(rows, t_ms)
    
    if before_idx is None or after_idx is None:
        return {"answer": "Unknown", "reasoning": "No valid timestamps"}

    for axis in joints_list:
        speed_key = f"feedback_speed_{axis}"
        val, _ = interpolate_signal_at_time(rows, speed_key, t_ms)
        if val is None:
             return {"answer": "Unknown", "reasoning": f"Missing speed data for joint {axis}"}
        speeds.append((axis, abs(val)))

    # Sort descending by speed
    speeds.sort(key=lambda x: x[1], reverse=True)
    
    # Ranked order corresponding to indices of joints_list 
    # For joints A, B, C, D maps to 0, 1, 2, 3
    # e.g., if joints_list = [2, 4, 1, 5] assigned labels A, B, C, D
    # and sorted speeds are joint 4, joint 2, joint 5, joint 1
    # order is B, A, D, C
    joint_to_label = {j: chr(ord('A') + i) for i, j in enumerate(joints_list)}
    
    ranking = "".join(joint_to_label[j] for j, _ in speeds)
    return {"answer": ranking, "reasoning": f"Speeds: {speeds}"}
