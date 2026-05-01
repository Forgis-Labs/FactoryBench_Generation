from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional

EPS = 1e-9
MC_IDS = [f"mc_{i:03d}" for i in range(1, 20)]


def _canonical_statement_id(statement_id: str) -> str:
    text = str(statement_id).strip()
    match = re.match(r"^(?:mc|l2_mc)_(\d+)$", text, re.IGNORECASE)
    if not match:
        return text
    return f"l2_mc_{int(match.group(1)):03d}"

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "speed_drop_ratio": 0.30,
    "speed_stable_tol": 0.10,
    "stall_current_increase": 0.25,
    "stall_speed_frac": 0.10,
    "force_low_increase": 0.20,
    "force_low_coverage": 0.80,
    "force_spike_increase": 0.40,
    "tracking_increase": 0.25,
    "tracking_stable_increase": 0.10,
    "vibration_spike": 0.50,
    "vibration_nominal_band": 0.15,
    "vibration_nominal_coverage": 0.90,
    "current_peak_increase": 0.20,
    "current_relax_drop": 0.10,
    "robot_current_stable_range": 0.10,
    "robot_current_increase": 0.20,
    "tcp_tracking_stable_increase": 0.10,
    "tcp_tracking_increase": 0.25,
    "temp_rise_slope": 0.005,
    "temp_rise_min_axes": 2.0,
    "temp_stable_slope": 0.002,
    "temp_stable_axes_ratio": 0.80,
    "no_effect_agg_increase": 0.10,
}


def _get_threshold(thresholds: Optional[Dict[str, float]], key: str) -> float:
    if thresholds and key in thresholds:
        try:
            return float(thresholds[key])
        except (TypeError, ValueError):
            pass
    return float(DEFAULT_THRESHOLDS[key])


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _baseline_row(subseries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not subseries:
        return None
    row = subseries[-1]
    return row if isinstance(row, dict) else None


def _series(post_event_rows: List[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in post_event_rows:
        if not isinstance(row, dict):
            continue
        value = _to_float(row.get(key))
        if value is not None and math.isfinite(value):
            values.append(value)
    return values


def _indexed_keys(prefix: str, rows: Iterable[Dict[str, Any]]) -> List[str]:
    keys: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            if isinstance(key, str) and key.startswith(prefix + "_"):
                tail = key[len(prefix) + 1 :]
                if tail.isdigit():
                    keys.add(key)
    return sorted(keys, key=lambda x: int(x.rsplit("_", 1)[1]))


def _safe_ratio(value: float, baseline: float) -> float:
    return value / max(EPS, abs(baseline))


def _vector_norm(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(math.sqrt(sum(v * v for v in values)))


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _tracking_error_row(row: Dict[str, Any], tcp: bool = False) -> Optional[float]:
    pos_prefix = "setpoint_tcp" if tcp else "setpoint_pos"
    fb_pos_prefix = "feedback_tcp" if tcp else "feedback_pos"
    spd_prefix = "setpoint_tcp_speed" if tcp else "setpoint_speed"
    fb_spd_prefix = "feedback_tcp_speed" if tcp else "feedback_speed"

    pos_keys = _indexed_keys(pos_prefix, [row])
    fb_pos_keys = _indexed_keys(fb_pos_prefix, [row])
    common_axes = []
    for key in pos_keys:
        axis = key.rsplit("_", 1)[1]
        fb_key = f"{fb_pos_prefix}_{axis}"
        if fb_key in fb_pos_keys:
            common_axes.append(axis)

    if not common_axes:
        return None

    errs: List[float] = []
    for axis in common_axes:
        sp = _to_float(row.get(f"{pos_prefix}_{axis}"))
        fb = _to_float(row.get(f"{fb_pos_prefix}_{axis}"))
        if sp is None or fb is None:
            continue
        err = abs(sp - fb)
        spd_sp = _to_float(row.get(f"{spd_prefix}_{axis}"))
        spd_fb = _to_float(row.get(f"{fb_spd_prefix}_{axis}"))
        if spd_sp is not None and spd_fb is not None:
            err += abs(spd_sp - spd_fb)
        errs.append(err)

    if not errs:
        return None
    return _mean(errs)


def _tracking_errors(rows: List[Dict[str, Any]], tcp: bool = False) -> List[float]:
    values: List[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _tracking_error_row(row, tcp=tcp)
        if value is not None:
            values.append(value)
    return values


def _contact_norm_row(row: Dict[str, Any]) -> Optional[float]:
    est_keys = _indexed_keys("est_contact_force", [row])
    true_keys = _indexed_keys("true_force", [row])
    use_keys = est_keys if est_keys else true_keys
    if not use_keys:
        return None

    vals: List[float] = []
    for key in use_keys:
        value = _to_float(row.get(key))
        if value is None:
            return None
        vals.append(value)
    return _vector_norm(vals)


def _contact_norms(rows: List[Dict[str, Any]]) -> List[float]:
    norms: List[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _contact_norm_row(row)
        if value is not None:
            norms.append(value)
    return norms


def _aggregate_error(rows: List[Dict[str, Any]], baseline_row: Optional[Dict[str, Any]]) -> Optional[float]:
    tr = _mean(_tracking_errors(rows, tcp=False))
    tcp_tr = _mean(_tracking_errors(rows, tcp=True))
    force_n = _mean(_contact_norms(rows))

    robot_term: Optional[float] = None
    if baseline_row is not None:
        pre_robot = _to_float(baseline_row.get("robot_current"))
        post_robot = _mean(_series(rows, "robot_current"))
        if pre_robot is not None and post_robot is not None:
            robot_term = abs(post_robot - pre_robot) / max(EPS, abs(pre_robot))

    terms = [x for x in [tr, tcp_tr, force_n, robot_term] if x is not None]
    if not terms:
        return None
    return float(sum(terms) / len(terms))


def evaluate_mc_statement(
    statement_id: str,
    subseries: List[Dict[str, Any]],
    post_event_rows: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
    episode_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[bool]:
    sid = _canonical_statement_id(statement_id)

    if sid == "l2_mc_020":
        if episode_metadata is None:
            return None
        success = episode_metadata.get("task_success")
        if success is None:
            return None
        return bool(success)

    baseline = _baseline_row(subseries)
    if baseline is None or not post_event_rows:
        return None

    if sid in {"l2_mc_003", "l2_mc_004", "l2_mc_005", "l2_mc_012"}:
        speed_keys = _indexed_keys("feedback_speed", [baseline])
        current_keys = _indexed_keys("effort_current", [baseline])

    if sid == "l2_mc_003":
        drop_ratio = _get_threshold(thresholds, "speed_drop_ratio")
        for speed_key in speed_keys:
            pre = _to_float(baseline.get(speed_key))
            series = _series(post_event_rows, speed_key)
            if pre is None or not series:
                continue
            if min(series) <= (1.0 - drop_ratio) * pre:
                return True
        return False

    if sid == "l2_mc_004":
        stable_tol = _get_threshold(thresholds, "speed_stable_tol")
        for speed_key in speed_keys:
            pre = _to_float(baseline.get(speed_key))
            series = _series(post_event_rows, speed_key)
            if pre is None or not series:
                continue
            max_abs_diff = max(abs(v - pre) for v in series)
            if max_abs_diff > stable_tol * max(EPS, abs(pre)):
                return False
        return True

    if sid == "l2_mc_005":
        current_increase = _get_threshold(thresholds, "stall_current_increase")
        speed_frac = _get_threshold(thresholds, "stall_speed_frac")
        for current_key in current_keys:
            axis = current_key.rsplit("_", 1)[1]
            speed_key = f"feedback_speed_{axis}"
            pre_current = _to_float(baseline.get(current_key))
            pre_speed = _to_float(baseline.get(speed_key))
            post_current = _mean(_series(post_event_rows, current_key))
            post_speed_abs = _mean([abs(v) for v in _series(post_event_rows, speed_key)])
            if (
                pre_current is None
                or pre_speed is None
                or post_current is None
                or post_speed_abs is None
            ):
                continue
            current_up = post_current >= (1.0 + current_increase) * pre_current
            speed_low = post_speed_abs <= speed_frac * max(EPS, abs(pre_speed))
            if current_up and speed_low:
                return True
        return False

    if sid == "l2_mc_006":
        low_increase = _get_threshold(thresholds, "force_low_increase")
        low_coverage = _get_threshold(thresholds, "force_low_coverage")
        pre_norm = _contact_norm_row(baseline)
        post_norms = _contact_norms(post_event_rows)
        if pre_norm is None or not post_norms:
            return None
        ok = sum(1 for n in post_norms if n <= (1.0 + low_increase) * pre_norm)
        return (ok / len(post_norms)) >= low_coverage

    if sid == "l2_mc_007":
        spike_increase = _get_threshold(thresholds, "force_spike_increase")
        pre_norm = _contact_norm_row(baseline)
        post_norms = _contact_norms(post_event_rows)
        if pre_norm is None or not post_norms:
            return None
        return any(n >= (1.0 + spike_increase) * pre_norm for n in post_norms)

    if sid in {"l2_mc_008", "l2_mc_009"}:
        tracking_increase = _get_threshold(thresholds, "tracking_increase")
        tracking_stable_increase = _get_threshold(thresholds, "tracking_stable_increase")
        pre_err = _tracking_error_row(baseline, tcp=False)
        post_err = _mean(_tracking_errors(post_event_rows, tcp=False))
        if pre_err is None or post_err is None:
            return None
        if sid == "l2_mc_008":
            return post_err >= (1.0 + tracking_increase) * pre_err
        return post_err <= (1.0 + tracking_stable_increase) * pre_err

    if sid == "l2_mc_012":
        peak_increase = _get_threshold(thresholds, "current_peak_increase")
        relax_drop = _get_threshold(thresholds, "current_relax_drop")
        split = max(1, int(0.4 * len(post_event_rows)))
        early = post_event_rows[:split]
        for current_key in current_keys:
            pre = _to_float(baseline.get(current_key))
            early_series = _series(early, current_key)
            post_series = _series(post_event_rows, current_key)
            if pre is None or not early_series or not post_series:
                continue
            peak = max(early_series)
            final = post_series[-1]
            if peak >= (1.0 + peak_increase) * pre and final <= (1.0 - relax_drop) * peak:
                return True
        return False

    if sid in {"l2_mc_013", "l2_mc_014"}:
        stable_range = _get_threshold(thresholds, "robot_current_stable_range")
        current_increase = _get_threshold(thresholds, "robot_current_increase")
        pre = _to_float(baseline.get("robot_current"))
        series = _series(post_event_rows, "robot_current")
        if pre is None or not series:
            return None
        mean_series = _mean(series)
        if mean_series is None:
            return None
        if sid == "l2_mc_013":
            return (max(series) - min(series)) <= stable_range * max(EPS, abs(pre))
        return mean_series >= (1.0 + current_increase) * pre

    if sid in {"l2_mc_015", "l2_mc_016"}:
        tcp_stable = _get_threshold(thresholds, "tcp_tracking_stable_increase")
        tcp_increase = _get_threshold(thresholds, "tcp_tracking_increase")
        pre_err = _tracking_error_row(baseline, tcp=True)
        post_err = _mean(_tracking_errors(post_event_rows, tcp=True))
        if pre_err is None or post_err is None:
            return None
        if sid == "l2_mc_015":
            return post_err <= (1.0 + tcp_stable) * pre_err
        return post_err >= (1.0 + tcp_increase) * pre_err

    if sid in {"l2_mc_017", "l2_mc_018"}:
        temp_rise_slope = _get_threshold(thresholds, "temp_rise_slope")
        temp_rise_min_axes = int(round(_get_threshold(thresholds, "temp_rise_min_axes")))
        temp_stable_slope = _get_threshold(thresholds, "temp_stable_slope")
        temp_stable_axes_ratio = _get_threshold(thresholds, "temp_stable_axes_ratio")
        temp_keys = _indexed_keys("joint_temp", [baseline])
        if not temp_keys:
            return None
        rises = 0
        stable = 0
        valid = 0
        for key in temp_keys:
            pre = _to_float(baseline.get(key))
            series = _series(post_event_rows, key)
            if pre is None or len(series) < 2:
                continue
            valid += 1
            slope = (series[-1] - series[0]) / max(EPS, abs(pre))
            if slope > temp_rise_slope:
                rises += 1
            if abs(slope) <= temp_stable_slope:
                stable += 1
        if valid == 0:
            return None
        if sid == "l2_mc_017":
            return rises >= temp_rise_min_axes
        return (stable / valid) >= temp_stable_axes_ratio

    if sid == "l2_mc_019":
        agg_increase = _get_threshold(thresholds, "no_effect_agg_increase")
        safety_normal = True
        for row in post_event_rows:
            v = _to_float(row.get("safety_mode"))
            if v is None or int(v) != 1:
                safety_normal = False
                break
        if not safety_normal:
            return False

        pre_agg = _aggregate_error([baseline], baseline)
        post_agg = _aggregate_error(post_event_rows, baseline)
        if pre_agg is None or post_agg is None:
            return None
        return post_agg <= (1.0 + agg_increase) * pre_agg

    return None


def evaluate_all_mc_statements(
    subseries: List[Dict[str, Any]],
    post_event_rows: List[Dict[str, Any]],
    statement_ids: Optional[List[str]] = None,
    thresholds_by_id: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Optional[bool]]:
    ids = statement_ids or MC_IDS
    return {
        sid: evaluate_mc_statement(
            sid,
            subseries=subseries,
            post_event_rows=post_event_rows,
            thresholds=(thresholds_by_id or {}).get(sid),
        )
        for sid in ids
    }
