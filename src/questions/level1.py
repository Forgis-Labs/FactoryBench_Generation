"""
Level 1 question generator: State identification.

Generates questions about robot state and computes deterministic ground-truth answers
following exact methodologies for:
  Q1: Joint position comparison
  Q2: Friction increase detection
  Q3: End-effector acceleration

Ground-truth is computed deterministically from the normalized JSON.
If required signals are missing, the ground-truth is set to null.
"""

from __future__ import annotations

import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def load_normalized_episode(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_phrases(path: Union[str, Path]) -> List[str]:
    p = Path(path)
    if not p.exists():
        # fallback to default single phrase
        return ["Has joint {axis} moved between {t1}ms and {t2}ms (threshold {eps_q} rad)?"]
    try:
        with open(p, "r", encoding="utf-8") as f:
            phrases = json.load(f)
        if isinstance(phrases, list) and all(isinstance(s, str) for s in phrases):
            return phrases
    except Exception:
        pass
    return ["Has joint {axis} moved between {t1}ms and {t2}ms (threshold of {eps_q} rad)?"]


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
    """
    Q1: Is joint at same position in t1 and t2 (threshold eps_1)?
    
    Returns dict with 'answer' (Yes/No/Unknown) and 'evidence'.
    """
    pos_key = f"feedback_pos_{axis}"
    
    # Find bracketing samples for t1 and t2
    before_t1, after_t1 = find_bracketing_indices(rows, t1_ms)
    before_t2, after_t2 = find_bracketing_indices(rows, t2_ms)
    
    if before_t1 is None or after_t1 is None or before_t2 is None or after_t2 is None:
        reasoning = (
            f"I define t1={t1_ms}ms and t2={t2_ms}ms. I looked for samples bracketing t1 and t2, but at least one "
            "endpoint had no surrounding timestamp samples, so I cannot interpolate joint position. "
            "Answer: Unknown."
        )
        return {"answer": "Unknown", "reasoning": reasoning}
    
    # Read positions at bracketing times
    try:
        if before_t1 == after_t1:
            # Exact sample at t1
            q_t1 = float(rows[before_t1][pos_key])
        else:
            # Interpolate
            v1 = float(rows[before_t1][pos_key])
            v2 = float(rows[after_t1][pos_key])
            t_before = float(rows[before_t1]["timestamp_ms"])
            t_after = float(rows[after_t1]["timestamp_ms"])
            q_t1 = interpolate_value(t1_ms, t_before, v1, t_after, v2)
        
        if before_t2 == after_t2:
            q_t2 = float(rows[before_t2][pos_key])
        else:
            v1 = float(rows[before_t2][pos_key])
            v2 = float(rows[after_t2][pos_key])
            t_before = float(rows[before_t2]["timestamp_ms"])
            t_after = float(rows[after_t2]["timestamp_ms"])
            q_t2 = interpolate_value(t2_ms, t_before, v1, t_after, v2)
        
    except (KeyError, TypeError, ValueError):
        reasoning = (
            f"I define t1={t1_ms}ms and t2={t2_ms}ms. I attempted to read the joint position q(t) for joint {axis} around "
            "t1 and t2, but the position values are missing or not numeric, so I cannot compute positions. Answer: Unknown."
        )
        return {"answer": "Unknown", "reasoning": reasoning}
    
    # Compute displacement
    delta_q = abs(q_t2 - q_t1)
    
    # Decide
    moved = delta_q > eps_1
    
    interpolation_t1 = "exact" if before_t1 == after_t1 else "interpolated"
    interpolation_t2 = "exact" if before_t2 == after_t2 else "interpolated"
    reasoning = (
        f"I define t1={t1_ms}ms and t2={t2_ms}ms. I computed the joint position q(t) for joint {axis}. "
        f"At t1 I used an {interpolation_t1} value q(t1)={q_t1:.6f} rad, and at t2 I used an {interpolation_t2} value "
        f"q(t2)={q_t2:.6f} rad. The displacement is Δq={delta_q:.6f} rad. With threshold eps_1={eps_1}, "
        f"this is {'>' if moved else '<='} the threshold, so the answer is {'Yes' if moved else 'No'}."
    )
    return {"answer": "Yes" if moved else "No", "reasoning": reasoning}


def answer_q2_friction_increase(
    rows: List[Dict[str, Any]],
    t1_ms: float,
    t2_ms: float,
    axis: int,
    eps_2: float,
    delta_1_ms: int = 500,
) -> Dict[str, Any]:
    """
    Q2: Has friction increased between t1 and t2?
    
    Estimates friction proxy from current/speed ratio in symmetric windows.
    """
    speed_key = f"feedback_speed_{axis}"
    current_key = f"effort_current_{axis}"
    
    # Define windows and clip to available timestamp range
    timestamps = [r.get("timestamp_ms") for r in rows if r.get("timestamp_ms") is not None]
    if not timestamps:
        return {
            "answer": "Unknown",
            "reasoning": "No valid timestamps are available to build windows, so friction cannot be estimated."
        }
    min_ts = min(timestamps)
    max_ts = max(timestamps)

    w1_start_raw = t1_ms - delta_1_ms
    w1_end_raw = t1_ms + delta_1_ms
    w2_start_raw = t2_ms - delta_1_ms
    w2_end_raw = t2_ms + delta_1_ms

    w1_start = max(min_ts, w1_start_raw)
    w1_end = min(max_ts, w1_end_raw)
    w2_start = max(min_ts, w2_start_raw)
    w2_end = min(max_ts, w2_end_raw)
    
    def compute_friction_proxy(w_start, w_end):
        """Compute median(|I|/|v|) for samples in window."""
        # Filter rows in window
        window_rows = [r for r in rows 
                      if r.get("timestamp_ms") is not None 
                      and w_start <= r["timestamp_ms"] <= w_end]
        
        if len(window_rows) < 3:
            return None, "Insufficient samples in window"
        
        speeds = []
        currents = []
        
        for i in range(0, len(window_rows)):
            try:
                v_curr = float(window_rows[i][speed_key])
                current = float(window_rows[i][current_key])
                speeds.append(v_curr)
                currents.append(current)
            except (KeyError, TypeError, ValueError):
                continue
        
        # Compute friction proxy: median(|I|/|v|)
        ratios = [abs(currents[i]) / abs(speeds[i]) for i in range(len(speeds)) if abs(speeds[i]) > 1e-6]
        
        if not ratios:
            return None, "No valid current/speed ratios"
        
        return np.median(ratios), None
    
    f1, err1 = compute_friction_proxy(w1_start, w1_end)
    f2, err2 = compute_friction_proxy(w2_start, w2_end)
    
    if f1 is None or f2 is None:
        reasoning = (
            f"I define t1={t1_ms}ms and t2={t2_ms}ms. I built symmetric windows around t1 and t2 and clipped them to the data range. "
            f"W1=[{w1_start:.1f},{w1_end:.1f}] ms (raw [{w1_start_raw:.1f},{w1_end_raw:.1f}]) and "
            f"W2=[{w2_start:.1f},{w2_end:.1f}] ms (raw [{w2_start_raw:.1f},{w2_end_raw:.1f}]). "
            "I then used all samples in each window to compute the friction proxy. "
            f"The friction proxy could not be computed (W1: {err1 or 'OK'}, W2: {err2 or 'OK'}), so the answer is Unknown."
        )
        return {"answer": "Unknown", "reasoning": reasoning}

    # Decide: friction increased if f2/f1 > (1 + eps_2/100)
    ratio = f2 / f1 if f1 != 0 else float('inf')
    threshold_ratio = 1.0 + (eps_2 / 100.0)
    increased = ratio > threshold_ratio
    
    reasoning = (
        f"I define t1={t1_ms}ms and t2={t2_ms}ms. I built symmetric windows around t1 and t2 and clipped them to the data range. "
        f"W1=[{w1_start:.1f},{w1_end:.1f}] ms (raw [{w1_start_raw:.1f},{w1_end_raw:.1f}]) and "
        f"W2=[{w2_start:.1f},{w2_end:.1f}] ms (raw [{w2_start_raw:.1f},{w2_end_raw:.1f}]). "
        "I used joint speed and motor current and used all samples in each window. "
        f"The friction proxies are f1={float(f1):.6f} and f2={float(f2):.6f} (median of |I| / |v|). "
        f"The ratio f2/f1={float(ratio):.6f}, and the threshold is 1 + eps_2 / 100 = {threshold_ratio:.6f} "
        f"for eps_2={eps_2}%. Since the ratio is {'>' if increased else '<='} the threshold, the answer is {'Yes' if increased else 'No'}."
    )
    return {"answer": "Yes" if increased else "No", "reasoning": reasoning}


def answer_q3_end_effector_accel(
    rows: List[Dict[str, Any]],
    t1_ms: float
) -> Dict[str, Any]:
    """
    Q3: What is the end effector acceleration at t1?
    
    Uses vibration_0,1,2 (in g) and interpolates to t1, then converts to m/s^2.
    """
    before_idx, after_idx = find_bracketing_indices(rows, t1_ms)
    
    if before_idx is None or after_idx is None:
        reasoning = (
            f"I define t1={t1_ms}ms. I looked for samples bracketing t1 to interpolate vibration, but none were found, "
            "so acceleration cannot be estimated. Answer: Unknown."
        )
        return {"answer": "Unknown", "reasoning": reasoning}

    try:
        if before_idx == after_idx:
            # Exact sample
            vib_0 = float(rows[before_idx]["vibration_0"])
            vib_1 = float(rows[before_idx]["vibration_1"])
            vib_2 = float(rows[before_idx]["vibration_2"])
        else:
            # Interpolate each axis
            t_before = float(rows[before_idx]["timestamp_ms"])
            t_after = float(rows[after_idx]["timestamp_ms"])
            
            v0_before = float(rows[before_idx]["vibration_0"])
            v0_after = float(rows[after_idx]["vibration_0"])
            vib_0 = interpolate_value(t1_ms, t_before, v0_before, t_after, v0_after)
            
            v1_before = float(rows[before_idx]["vibration_1"])
            v1_after = float(rows[after_idx]["vibration_1"])
            vib_1 = interpolate_value(t1_ms, t_before, v1_before, t_after, v1_after)
            
            v2_before = float(rows[before_idx]["vibration_2"])
            v2_after = float(rows[after_idx]["vibration_2"])
            vib_2 = interpolate_value(t1_ms, t_before, v2_before, t_after, v2_after)
        
        # Convert from g to m/s^2
        G = 9.81
        accel_x = vib_0 * G
        accel_y = vib_1 * G
        accel_z = vib_2 * G
        
        magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        interp_mode = "exact" if before_idx == after_idx else "interpolated"
        reasoning = (
            f"I define t1={t1_ms}ms and computed {interp_mode} three-axis accelerometer values "
            f"v=[{float(vib_0):.6f}, {float(vib_1):.6f}, {float(vib_2):.6f}] g. "
            f"Converting with g=9.81 m/s² gives a=[{float(accel_x):.6f}, {float(accel_y):.6f}, {float(accel_z):.6f}] m/s², "
            f"with magnitude {float(magnitude):.6f} m/s²."
        )
        return {"answer": [accel_x, accel_y, accel_z], "reasoning": reasoning}
        
    except (KeyError, TypeError, ValueError) as e:
        reasoning = (
            f"I define t1={t1_ms}ms. I attempted to read the three-axis accelerometer values around t1 but encountered missing or non-numeric values ({e}). "
            "Answer: Unknown."
        )
        return {"answer": "Unknown", "reasoning": reasoning}


def answer_q4_external_force(
    rows: List[Dict[str, Any]],
    t_ms: float,
    eps_3: float,
) -> Dict[str, Any]:
    """Q4: Is external force detected at time T?"""
    values, source, interp_mode = get_wrench_components_at_time(rows, t_ms, [0, 1, 2])
    if values is None:
        reasoning = (
            f"I define t={t_ms}ms. I tried to get the three force components at t using the external sensor first and then the controller estimate, "
            "but the values were missing, so I cannot compute force magnitude. Answer: Unknown."
        )
        return {"answer": "Unknown", "reasoning": reasoning}

    fx, fy, fz = values
    magnitude = np.sqrt(fx**2 + fy**2 + fz**2)
    detected = magnitude >= eps_3
    reasoning = (
        f"I define t={t_ms}ms. I obtained the force components at t using the {source} with {interp_mode} values: "
        f"Fx={fx:.6f} N, Fy={fy:.6f} N, Fz={fz:.6f} N. The magnitude is F=√(Fx² + Fy² + Fz²)={magnitude:.6f} N. "
        f"The threshold eps_3={eps_3} N. Since F is {'>=' if detected else '<'} eps_3, "
        f"the answer is {'Yes' if detected else 'No'}."
    )
    return {"answer": "Yes" if detected else "No", "reasoning": reasoning}


def answer_q5_joint_jerk(
    rows: List[Dict[str, Any]],
    t_ms: float,
    axis: int,
) -> Dict[str, Any]:
    """Q5: What is the jerk of joint i at time T?"""
    valid = [(i, r.get("timestamp_ms")) for i, r in enumerate(rows) if r.get("timestamp_ms") is not None]
    if len(valid) < 5:
        reasoning = "There are not enough timestamped samples to estimate jerk, so the answer is Unknown."
        return {"answer": "Unknown", "reasoning": reasoning}

    # Find closest timestamp to t_ms
    closest_pos = min(range(len(valid)), key=lambda p: abs(valid[p][1] - t_ms))
    if closest_pos < 2 or closest_pos + 2 >= len(valid):
        reasoning = (
            f"I define t={t_ms}ms. The closest timestamp to t does not have two samples on each side, so jerk cannot be estimated. "
            "Answer: Unknown."
        )
        return {"answer": "Unknown", "reasoning": reasoning}

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
            raise ValueError("Non-positive time differences")

        accel_km1 = (v_k - v_km2) / dt_k_km2
        accel_kp1 = (v_kp2 - v_k) / dt_kp2_k

        dt_kp1_km1 = (t_kp1 - t_km1) / 1000.0
        if dt_kp1_km1 <= 0:
            raise ValueError("Non-positive time differences")

        jerk = (accel_kp1 - accel_km1) / dt_kp1_km1
    except (KeyError, TypeError, ValueError):
        reasoning = (
            f"I define t={t_ms}ms. I attempted to compute jerk for joint {axis} at t, but required velocity samples were missing or invalid. "
            "Answer: Unknown."
        )
        return {"answer": "Unknown", "reasoning": reasoning}

    reasoning = (
        f"I define t={t_ms}ms. I found the closest timestamp to t at {t_k}ms and used two samples on each side at "
        f"{t_km2}ms, {t_km1}ms, {t_kp1}ms, and {t_kp2}ms. Any common derivative estimation is acceptable; here I use central differences. "
        f"I computed a₋={accel_km1:.6f} rad/s² = (v_k - v_km2) / Δt_-, and a₊={accel_kp1:.6f} rad/s² = (v_kp2 - v_k) / Δt_+. "
        f"Then jerk j = (a₊ - a₋) / Δt = {jerk:.6f} rad/s³."
    )
    return {"answer": jerk, "reasoning": reasoning}


def answer_q6_torque_magnitude(
    rows: List[Dict[str, Any]],
    t_ms: float,
    axis_label: str,
) -> Dict[str, Any]:
    """Q6: What is the torque magnitude about axis at time T?"""
    axis_map = {"x": 3, "y": 4, "z": 5}
    if axis_label not in axis_map:
        return {"answer": "Unknown", "reasoning": "Axis label is invalid, so torque cannot be computed."}

    idx = axis_map[axis_label]
    values, source, interp_mode = get_wrench_components_at_time(rows, t_ms, [idx])
    if values is None:
        reasoning = (
            f"I define t={t_ms}ms. I tried to get the torque component about axis {axis_label} at t using the external sensor first "
            "and then the controller estimate, but the value was missing, so the answer is Unknown."
        )
        return {"answer": "Unknown", "reasoning": reasoning}

    torque = values[0]
    magnitude = abs(torque)
    reasoning = (
        f"I define t={t_ms}ms. I obtained the torque component about axis {axis_label} at t using the {source} with {interp_mode} values: "
        f"τ={torque:.6f} Nm, |τ|={magnitude:.6f} Nm."
    )
    return {"answer": magnitude, "reasoning": reasoning}


def pick_time_window(rows: List[Dict[str, Any]], min_dt_ms: int, max_dt_ms: int) -> Tuple[int, int]:
    n = len(rows)
    if n < 2:
        raise ValueError("Not enough samples to pick a time window")
    # Build list of indices that have valid timestamps
    valid_idx = [i for i, r in enumerate(rows) if r.get("timestamp_ms") is not None]
    if len(valid_idx) < 2:
        # fallback to endpoints
        return 0, n - 1

    attempts = 0
    while attempts < 1000:
        i_pos = random.randint(0, len(valid_idx) - 2)
        j_pos = random.randint(i_pos + 1, len(valid_idx) - 1)
        i = valid_idx[i_pos]
        j = valid_idx[j_pos]
        dt = rows[j]["timestamp_ms"] - rows[i]["timestamp_ms"]
        if dt is None:
            attempts += 1
            continue
        if min_dt_ms <= dt <= max_dt_ms:
            return i, j
        attempts += 1

    # fallback: return first and last valid indices
    return valid_idx[0], valid_idx[-1]


def generate_level1_questions(
    episode_json: Path,
    out_json: Path,
    n_questions: int = 100,
    min_dt_ms: int = 100,
    max_dt_ms: int = 2000,
    eps_q: float = 1e-3,
    seed: Optional[int] = None,
    phrases_file: Optional[Path] = None,
    eps_1: float = None,
    eps_2: float = None,
    delta_1: int = None,
    eps_3: float = None,
) -> List[Dict[str, Any]]:
    """Generate Level 1 questions for a single normalized episode.

    Args:
        episode_json: path to normalized episode JSON (list of rows)
        out_json: where to write generated questions (JSON array)
        n_questions: number of questions to generate
        min_dt_ms, max_dt_ms: time-window size bounds (ms)
        eps_q: threshold (rad) for determining "moved" (legacy, used if eps_1/eps_2 not set)
        seed: RNG seed
        phrases_file: path to phrases JSON file
        eps_1: threshold (rad) for question 1 (same position check)
        eps_2: threshold (percent) for question 2 (friction increase check)
        delta_1: time window (ms) for question 2 (symmetric half window)
        eps_3: threshold (N) for question 4 (external force detection)
    Returns:
        List of question dicts written to `out_json`.
    """
    if seed is not None:
        random.seed(seed)

    # Set defaults for new parameters
    if eps_1 is None:
        eps_1 = eps_q
    if eps_2 is None:
        eps_2 = eps_q
    if delta_1 is None:
        delta_1 = max_dt_ms // 2
    if eps_3 is None:
        eps_3 = eps_q

    rows = load_normalized_episode(episode_json)
    # load phrase templates
    phrases = load_phrases(phrases_file or Path(__file__).with_name("phrases_level1.json"))
    n_rows = len(rows)
    if n_rows == 0:
        raise ValueError("Episode JSON contains no rows")

    questions: List[Dict[str, Any]] = []

    for qid in range(1, n_questions + 1):
        axis = random.randint(0, 5)
        start_idx, end_idx = pick_time_window(rows, min_dt_ms, max_dt_ms)
        t1 = rows[start_idx]["timestamp_ms"]
        t2 = rows[end_idx]["timestamp_ms"]
        t = t1
        axis_label = random.choice(["x", "y", "z"])

        # Pick a random phrase template
        template_idx = random.randint(0, len(phrases) - 1)
        template = phrases[template_idx]
        
        # Render question text
        try:
            text = template.format(
                axis=axis,
                t1=t1,
                t2=t2,
                t=t,
                eps_q=eps_q,
                eps_1=eps_1,
                eps_2=eps_2,
                eps_3=eps_3,
                Delta_1=delta_1,
                axis_label=axis_label,
            )
        except KeyError:
            text = f"Has joint {axis} moved between {t1}ms and {t2}ms (threshold of {eps_q} rad)?"
        
        # Compute answer based on question type (template index)
        if template_idx == 0:
            # Q1: Position check
            result = answer_q1_position_check(rows, t1, t2, axis, eps_1)
        elif template_idx == 1:
            # Q2: Friction increase
            result = answer_q2_friction_increase(
                rows,
                t1,
                t2,
                axis,
                eps_2,
                delta_1,
            )
        elif template_idx == 2:
            # Q3: End-effector acceleration
            result = answer_q3_end_effector_accel(rows, t1)
        elif template_idx == 3:
            # Q4: External force detection
            result = answer_q4_external_force(rows, t, eps_3)
        elif template_idx == 4:
            # Q5: Joint jerk
            result = answer_q5_joint_jerk(rows, t, axis)
        elif template_idx == 5:
            # Q6: Torque magnitude about axis
            result = answer_q6_torque_magnitude(rows, t, axis_label)
        else:
            # Fallback for unknown templates
            result = {"answer": "Unknown", "reasoning": "Unknown question type, so no answer was computed."}

        if result is None:
            result = {"answer": "Unknown", "reasoning": "Answer computation returned no result, so the answer is Unknown."}
        
        q = {
            "id": f"L1-{episode_json.stem}-{qid:04d}",
            "question": {"text": text, "level": 1, "type": template_idx + 1},
            "context": {
                "episode": str(episode_json),
                "time_window": [t1, t2],
                "joint": axis,
                "template_index": template_idx
            },
            "params": {
                "eps_1": eps_1,
                "eps_2": eps_2,
                "delta_1": delta_1,
                "eps_3": eps_3,
            },
            "answer": result["answer"],
            "reasoning": result.get("reasoning", ""),
            "provenance": "deterministic",
        }

        questions.append(q)

    # Write output file
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)

    logger.info(f"Wrote {len(questions)} Level 1 questions to {out_json}")
    return questions


if __name__ == "__main__":
    # lightweight CLI for quick runs
    import argparse

    parser = argparse.ArgumentParser(description="Generate Level 1 questions from normalized UR3e JSON.")
    parser.add_argument("--input", type=Path, required=True, help="Normalized episode JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output questions JSON file")
    parser.add_argument("--n", type=int, default=100, help="Number of questions to generate")
    parser.add_argument("--min-dt-ms", type=int, default=100, help="Minimum time window (ms)")
    parser.add_argument("--max-dt-ms", type=int, default=2000, help="Maximum time window (ms)")
    parser.add_argument("--eps-q", type=float, default=1e-3, help="Movement threshold in radians (legacy)")
    parser.add_argument("--eps-1", type=float, default=None, help="Threshold for Q1 (same position check, rad)")
    parser.add_argument("--eps-2", type=float, default=None, help="Threshold for Q2 (friction check, rad)")
    parser.add_argument("--delta-1", type=int, default=None, help="Time window for Q2 (symmetric half window, ms)")
    parser.add_argument("--eps-3", type=float, default=None, help="Threshold for Q4 external force detection")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    generate_level1_questions(
        args.input,
        args.output,
        n_questions=args.n,
        min_dt_ms=args.min_dt_ms,
        max_dt_ms=args.max_dt_ms,
        eps_q=args.eps_q,
        seed=args.seed,
        eps_1=args.eps_1,
        eps_2=args.eps_2,
        delta_1=args.delta_1,
        eps_3=args.eps_3,
    )
