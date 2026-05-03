"""
Level 2 question generator: Intervention Reasoning.

Reads normalized episode JSON files from aursad or vorausad datasets,
samples random sub-series, and fills Level 2 question templates.
Answers are generated when determinable from episode readings.

Output: datasets/questions/level2/level2_{NNNN}.json

Usage:
    python -m src.questions.level2.level2 -n 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from src.question_generation.utils.hf_streaming import (
    HfStreamUploader,
    add_streaming_args,
    list_completed_combos,
    make_uploader_from_args,
)
from src.question_generation.utils.io import load_events, load_json, load_root_causes, load_templates
from src.question_generation.utils.template import (
    build_context,
    discover_episodes_by_dataset,
    encode_chunk,
    fill,
    fill_event_description,
    get_last_timestamp,
    pick_constrained_signal,
    pick_scalar_signal,
)
from src.question_generation.utils.time_series import (
    parse_event_id,
    pick_fault_label,
    pick_fault_label_from_meta_or_rows,
    sample_subseries_before_event,
)
from src.question_generation.level2.mc_truth import DEFAULT_THRESHOLDS, evaluate_mc_statement
from src.question_generation.level1.level1 import (
    build_anomaly_single_select as l1_build_anomaly_single_select,
    build_comparative_multi_select as l1_build_comparative_multi_select,
    build_severity_ranking as l1_build_severity_ranking,
    get_severity_rank as l1_get_severity_rank,
    load_anomaly_lookup as l1_load_anomaly_lookup,
    load_anomaly_ranking as l1_load_anomaly_ranking,
    pick_phase_isolation_candidates as l1_pick_phase_isolation_candidates,
    _phase_display_name,
    _signal_display_name,
    PHASE_NAMES,
    DATASET_MACHINE_ID,
)

logger = logging.getLogger(__name__)

# Template IDs that use L1-style logic on anomalous data (not the event-based pipeline)
_L1_STYLE_TEMPLATE_IDS = {6, 7, 8, 9, 10}

# Human-readable anomaly phrases that fit mid-sentence after "suffers from"
_ANOMALY_INLINE_NAMES: Dict[int, str] = {
    1: "a damaged screw thread",
    2: "an extra assembly component in the workspace",
    3: "a missing screw",
    4: "a damaged plate thread",
    5: "a loosening phase instead of tightening",
    8: "a gripper activation failure",
    9: "a gripper release during motion",
    10: "additional payload on one of its axes",
    11: "a collision with a soft foam object",
    12: "an unexpected payload weight",
    14: "an invalid gripping position",
    15: "an unstable mounting platform",
    19: "a joint position limit violation",
    22: "a TCP frame misconfiguration",
    23: "a payload weight misconfiguration",
    25: "an external arm disturbance",
    28: "a payload center-of-gravity misconfiguration",
    29: "a collision with a hanging cable",
    30: "a collision with a cardboard object",
    31: "a collision with a rigid object",
    32: "a peg insertion misalignment",
    33: "a hole obstruction",
    34: "an incorrect insertion depth",
    35: "peg surface contamination",
    36: "a fixture displacement",
    37: "a self-collision between arm links",
    38: "a missing box at the pick position",
    39: "a missing peg",
}


def _anomaly_inline_name(fault_id: int, root_cause: Dict[str, Any], capitalize: bool = False) -> str:
    """Return a human-readable anomaly phrase that fits mid-sentence."""
    name = _ANOMALY_INLINE_NAMES.get(fault_id)
    if not name:
        name = root_cause.get("root_cause", f"fault {fault_id}").replace("_", " ")
        # Add article if missing
        if not name.startswith(("a ", "an ")):
            name = f"a {name}"
    if capitalize:
        return name[0].upper() + name[1:]
    return name

VALID_DATASETS = ["aursad", "vorausad", "factorywave", "factorywave_kuka"]

STEPS_AHEAD_RANGE = (1, 10)
CONTEXT_MIN = 32
CONTEXT_MAX = 64



EXCLUDED_JOINT_SIGNALS = {"joint_voltage", "joint_temp", "joint_mode"}
JOINT_INDEX_RANGE = set(range(6))
MIN_POST_EVENT_TIMESTAMPS_AFTER = 5

# MC option IDs that require signals absent from simulation data
NON_SIMULATION_EXCLUDED_MC_IDS = {
    "mc_020",  # task_success — only available in simulation metadata
}

SIMULATION_EXCLUDED_MC_IDS = {
    "mc_005",  # effort_current
    "mc_013",  # robot_current
    "mc_014",  # robot_current
    "mc_017",  # joint_temp
    "mc_019",  # safety_mode
}

# MC option IDs that use mode signals (safety_mode, joint_mode, robot_mode),
# excluded from predictive templates where mode state is not being predicted.
PREDICTIVE_EXCLUDED_MC_IDS = {
    "mc_019",  # safety_mode
}


def pick_joint_indexed_signal_base(subseries: List[Dict[str, Any]], allowed_bases: Optional[set] = None) -> Optional[str]:
    """
    Pick a base signal name that has indexed variants for all joints 0..5.

    Example valid base: "setpoint_pos" (requires setpoint_pos_0 ... setpoint_pos_5).
    Excludes: joint_voltage, joint_temp, joint_mode.
    If allowed_bases is provided, only bases in that set are considered.
    """
    base_to_indices: Dict[str, set[int]] = {}

    for row in subseries:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            match = re.match(r"^(.*)_(\d+)$", str(key))
            if not match:
                continue

            base = match.group(1)
            idx = int(match.group(2))
            if idx not in JOINT_INDEX_RANGE:
                continue

            if base not in base_to_indices:
                base_to_indices[base] = set()
            base_to_indices[base].add(idx)

    candidates = [
        base
        for base, indices in base_to_indices.items()
        if indices == JOINT_INDEX_RANGE and base not in EXCLUDED_JOINT_SIGNALS
        and (allowed_bases is None or base in allowed_bases)
    ]

    if not candidates:
        return None

    return random.choice(sorted(candidates))


def sample_subsequent_chunks(
    rows: List[Dict[str, Any]],
    n_chunks: int = 4,
    min_chunk: int = 5,
    max_chunk: int = 7,
) -> List[List[Dict[str, Any]]]:
    """
    Sample n_chunks contiguous, subsequent chunks from rows.

    Chunks are back-to-back in time (no gaps / no overlap), all with the same
    randomly chosen chunk length in [min_chunk, max_chunk].
    """
    available = len(rows)
    if available < n_chunks * min_chunk:
        return []

    max_feasible_chunk = min(max_chunk, available // n_chunks)
    if max_feasible_chunk < min_chunk:
        return []

    chunk_len = random.randint(min_chunk, max_feasible_chunk)
    total_len = n_chunks * chunk_len
    start = random.randint(0, available - total_len)

    chunks: List[List[Dict[str, Any]]] = []
    for i in range(n_chunks):
        left = start + i * chunk_len
        right = left + chunk_len
        chunks.append(rows[left:right])
    return chunks


def encode_chunk_without_timestamps(rows: List[Dict[str, Any]]) -> str:
    """Encode a chunk after removing timestamp_ms from each row."""
    stripped_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        stripped_rows.append({k: v for k, v in row.items() if k != "timestamp_ms"})
    return encode_chunk(stripped_rows)


def get_row_at_or_after_timestamp(
    rows: List[Dict[str, Any]],
    target_timestamp_ms: int,
) -> Optional[Dict[str, Any]]:
    """Return the first row whose timestamp_ms is >= target_timestamp_ms."""
    for row in rows:
        ts = row.get("timestamp_ms")
        try:
            if ts is not None and int(float(ts)) >= target_timestamp_ms:
                return row
        except (TypeError, ValueError):
            continue
    return None


def _first_timestamp_ms(rows: List[Dict[str, Any]]) -> int:
    """Return first valid timestamp_ms in rows, or 0 if missing."""
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts = row.get("timestamp_ms")
        try:
            if ts is not None:
                return int(float(ts))
        except (TypeError, ValueError):
            continue
    return 0


def normalize_timestamps(
    rows: List[Dict[str, Any]],
    base_timestamp_ms: int,
) -> List[Dict[str, Any]]:
    """Return a copy of rows with timestamp_ms shifted by base_timestamp_ms."""
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out_row = dict(row)
        ts = out_row.get("timestamp_ms")
        try:
            if ts is not None:
                out_row["timestamp_ms"] = int(float(ts)) - int(base_timestamp_ms)
        except (TypeError, ValueError):
            pass
        normalized.append(out_row)
    return normalized


def split_event_segment(
    post_event_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split rows starting at event onset into:
    - full contiguous event segment (same non-zero event token)
    - tail rows after the event segment
    """
    if not post_event_rows:
        return [], []

    onset_raw = post_event_rows[0].get("event", 0)
    onset_id = parse_event_id(onset_raw)
    if onset_id == 0:
        return [], post_event_rows

    event_segment: List[Dict[str, Any]] = []
    end_idx = 0
    onset_token = str(onset_raw)

    for i, row in enumerate(post_event_rows):
        row_token = str(row.get("event", 0))
        row_id = parse_event_id(row.get("event", 0))
        if row_id != onset_id or row_token != onset_token:
            end_idx = i
            break
        event_segment.append(row)
    else:
        end_idx = len(post_event_rows)

    return event_segment, post_event_rows[end_idx:]


def is_escalated_by_safety_mode(rows: List[Dict[str, Any]]) -> bool:
    """
    Escalation rule for template 3:
    If any timestep in the remaining sample has safety_mode != 1, it escalated.
    """
    for row in rows:
        value = row.get("safety_mode")
        try:
            if value is not None and int(float(value)) != 1:
                return True
        except (TypeError, ValueError):
            continue
    return False


def normalize_mc_option_id(value: str) -> str:
    """Normalize IDs like mc_19/l2_mc_19 -> mc_019."""
    text = str(value).strip()
    match = re.match(r"^(?:mc|l2_mc)_(\d+)$", text, re.IGNORECASE)
    if not match:
        return text
    return f"mc_{int(match.group(1)):03d}"


def _legacy_mc_option_id(value: str) -> str:
    """Convert normalized mc_* IDs to legacy l2_mc_* IDs used by current rules."""
    normalized = normalize_mc_option_id(value)
    match = re.match(r"^mc_(\d+)$", normalized, re.IGNORECASE)
    if not match:
        return normalized
    return f"l2_mc_{int(match.group(1)):03d}"


def load_mc_option_lookup(path: Path, level: int) -> Dict[str, str]:
    """
    Load MC options as a map: id -> statement.
    Returns empty map if file is missing/invalid.
    """
    try:
        raw = load_json(path)
    except Exception as exc:
        logger.warning(f"Could not load MC options from {path}: {exc}")
        return {}

    if not isinstance(raw, list):
        logger.warning(f"MC options file has unexpected format (expected list): {path}")
        return {}

    lookup: Dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue

        allowed_levels = item.get("usable_levels")
        if isinstance(allowed_levels, list):
            parsed_levels: List[int] = []
            for value in allowed_levels:
                try:
                    parsed_levels.append(int(value))
                except (TypeError, ValueError):
                    continue
            if parsed_levels and level not in parsed_levels:
                continue
        else:
            legacy_level = item.get("applicable_level")
            if legacy_level is not None:
                try:
                    if int(legacy_level) != level:
                        continue
                except (TypeError, ValueError):
                    continue

        option_id = item.get("id")
        statement = item.get("statement")
        if isinstance(option_id, str) and isinstance(statement, str) and statement.strip():
            lookup[normalize_mc_option_id(option_id)] = statement.strip()
    return lookup


def resolve_fixed_option(
    token: Any,
    mc_option_lookup: Dict[str, str],
) -> Tuple[Optional[str], str]:
    """
    Resolve a fixed option token to (canonical_option_id, rendered_statement).
    If token is not a known MC ID, returns (None, str(token)).
    """
    if isinstance(token, str):
        canonical_id = normalize_mc_option_id(token)
        statement = mc_option_lookup.get(canonical_id)
        if statement:
            return canonical_id, statement
        return None, token
    return None, str(token)


def _sample_ratio(mean: float, rel_std: float = 0.20, min_value: float = 0.0, max_value: float = 0.99) -> float:
    sampled = random.gauss(mean, max(1e-6, abs(mean) * rel_std))
    return float(min(max(sampled, min_value), max_value))


def sample_thresholds_for_statement(statement_id: str) -> Dict[str, float]:
    sid = _legacy_mc_option_id(str(statement_id))
    if sid == "l2_mc_003":
        return {"speed_drop_ratio": _sample_ratio(DEFAULT_THRESHOLDS["speed_drop_ratio"])}
    if sid == "l2_mc_004":
        return {"speed_stable_tol": _sample_ratio(DEFAULT_THRESHOLDS["speed_stable_tol"])}
    if sid == "l2_mc_005":
        return {
            "stall_current_increase": _sample_ratio(DEFAULT_THRESHOLDS["stall_current_increase"]),
            "stall_speed_frac": _sample_ratio(DEFAULT_THRESHOLDS["stall_speed_frac"]),
        }
    if sid == "l2_mc_006":
        return {
            "force_low_increase": _sample_ratio(DEFAULT_THRESHOLDS["force_low_increase"]),
            "force_low_coverage": _sample_ratio(DEFAULT_THRESHOLDS["force_low_coverage"], rel_std=0.08, min_value=0.50, max_value=0.99),
        }
    if sid == "l2_mc_007":
        return {"force_spike_increase": _sample_ratio(DEFAULT_THRESHOLDS["force_spike_increase"])}
    if sid == "l2_mc_008":
        return {"tracking_increase": _sample_ratio(DEFAULT_THRESHOLDS["tracking_increase"])}
    if sid == "l2_mc_009":
        return {"tracking_stable_increase": _sample_ratio(DEFAULT_THRESHOLDS["tracking_stable_increase"])}
    if sid == "l2_mc_010":
        return {"vibration_spike": _sample_ratio(DEFAULT_THRESHOLDS["vibration_spike"])}
    if sid == "l2_mc_011":
        return {
            "vibration_nominal_band": _sample_ratio(DEFAULT_THRESHOLDS["vibration_nominal_band"]),
            "vibration_nominal_coverage": _sample_ratio(DEFAULT_THRESHOLDS["vibration_nominal_coverage"], rel_std=0.06, min_value=0.60, max_value=0.99),
        }
    if sid == "l2_mc_012":
        return {
            "current_peak_increase": _sample_ratio(DEFAULT_THRESHOLDS["current_peak_increase"]),
            "current_relax_drop": _sample_ratio(DEFAULT_THRESHOLDS["current_relax_drop"]),
        }
    if sid == "l2_mc_013":
        return {"robot_current_stable_range": _sample_ratio(DEFAULT_THRESHOLDS["robot_current_stable_range"])}
    if sid == "l2_mc_014":
        return {"robot_current_increase": _sample_ratio(DEFAULT_THRESHOLDS["robot_current_increase"])}
    if sid == "l2_mc_015":
        return {"tcp_tracking_stable_increase": _sample_ratio(DEFAULT_THRESHOLDS["tcp_tracking_stable_increase"])}
    if sid == "l2_mc_016":
        return {"tcp_tracking_increase": _sample_ratio(DEFAULT_THRESHOLDS["tcp_tracking_increase"])}
    if sid == "l2_mc_017":
        min_axes = int(round(random.gauss(DEFAULT_THRESHOLDS["temp_rise_min_axes"], 0.4)))
        min_axes = min(max(min_axes, 1), 6)
        return {
            "temp_rise_slope": _sample_ratio(DEFAULT_THRESHOLDS["temp_rise_slope"], rel_std=0.25, min_value=0.0005, max_value=0.02),
            "temp_rise_min_axes": float(min_axes),
        }
    if sid == "l2_mc_018":
        return {
            "temp_stable_slope": _sample_ratio(DEFAULT_THRESHOLDS["temp_stable_slope"], rel_std=0.25, min_value=0.0002, max_value=0.01),
            "temp_stable_axes_ratio": _sample_ratio(DEFAULT_THRESHOLDS["temp_stable_axes_ratio"], rel_std=0.08, min_value=0.50, max_value=0.99),
        }
    if sid == "l2_mc_019":
        return {"no_effect_agg_increase": _sample_ratio(DEFAULT_THRESHOLDS["no_effect_agg_increase"])}
    return {}


def _fmt_pct(value: float) -> str:
    return str(int(round(100.0 * value)))


def render_statement_with_thresholds(
    statement_id: str,
    default_statement: str,
    thresholds: Dict[str, float],
) -> str:
    sid = _legacy_mc_option_id(str(statement_id))
    if sid == "l2_mc_003":
        return (
            "Following the event, at least one joint speed drops sharply "
            f"(>={_fmt_pct(thresholds['speed_drop_ratio'])}% below pre-event baseline)."
        )
    if sid == "l2_mc_004":
        return (
            "Following the event, joint speeds stay close to baseline "
            f"(within ±{_fmt_pct(thresholds['speed_stable_tol'])}% of pre-event values)."
        )
    if sid == "l2_mc_005":
        return (
            "Following the event, motor current rises by "
            f">={_fmt_pct(thresholds['stall_current_increase'])}% while speed magnitude stays "
            f"<={_fmt_pct(thresholds['stall_speed_frac'])}% of pre-event baseline (stall-like behavior)."
        )
    if sid == "l2_mc_006":
        return (
            "Following the event, contact-force magnitude remains low "
            f"(<={_fmt_pct(thresholds['force_low_increase'])}% above pre-event baseline for at least "
            f"{_fmt_pct(thresholds['force_low_coverage'])}% of timesteps)."
        )
    if sid == "l2_mc_007":
        return (
            "Following the event, contact force shows a significant spike "
            f"(>={_fmt_pct(thresholds['force_spike_increase'])}% above baseline norm)."
        )
    if sid == "l2_mc_008":
        return (
            "Following the event, tracking error increases noticeably "
            f"(>={_fmt_pct(thresholds['tracking_increase'])}% above pre-event mean)."
        )
    if sid == "l2_mc_009":
        return (
            "Following the event, tracking error is stable or improved "
            f"(increase <={_fmt_pct(thresholds['tracking_stable_increase'])}% vs pre-event mean)."
        )
    if sid == "l2_mc_010":
        return (
            "Following the event, vibration exhibits a transient burst "
            f"(>={_fmt_pct(thresholds['vibration_spike'])}% above pre-event baseline on at least one axis)."
        )
    if sid == "l2_mc_011":
        return (
            "Following the event, vibration remains nominal "
            f"(within ±{_fmt_pct(thresholds['vibration_nominal_band'])}% for at least "
            f"{_fmt_pct(thresholds['vibration_nominal_coverage'])}% of timesteps)."
        )
    if sid == "l2_mc_012":
        return (
            "Following the event, at least one joint current peaks then relaxes "
            f"(early peak >={_fmt_pct(thresholds['current_peak_increase'])}% above baseline, "
            f"final value >={_fmt_pct(thresholds['current_relax_drop'])}% below that peak)."
        )
    if sid == "l2_mc_013":
        return (
            "Following the event, robot current remains approximately constant "
            f"(peak-to-peak range <={_fmt_pct(thresholds['robot_current_stable_range'])}% of pre-event baseline)."
        )
    if sid == "l2_mc_014":
        return (
            "Following the event, robot current increases markedly "
            f"(>={_fmt_pct(thresholds['robot_current_increase'])}% above pre-event mean)."
        )
    if sid == "l2_mc_015":
        return (
            "Following the event, command and measured TCP motion remain aligned "
            f"(TCP tracking error increase <={_fmt_pct(thresholds['tcp_tracking_stable_increase'])}%)."
        )
    if sid == "l2_mc_016":
        return (
            "Following the event, command and measured TCP motion become misaligned "
            f"(TCP tracking error increase >={_fmt_pct(thresholds['tcp_tracking_increase'])}%)."
        )
    if sid == "l2_mc_017":
        return (
            "Following the event, temperatures rise across multiple joints "
            f"(>={int(round(thresholds['temp_rise_min_axes']))} joints with slope >"
            f"{100.0 * thresholds['temp_rise_slope']:.2f}% of baseline per horizon)."
        )
    if sid == "l2_mc_018":
        return (
            "Following the event, temperatures remain stable "
            f"(slope magnitude <={100.0 * thresholds['temp_stable_slope']:.2f}% of baseline per horizon "
            f"for at least {_fmt_pct(thresholds['temp_stable_axes_ratio'])}% of joints)."
        )
    if sid == "l2_mc_019":
        return (
            "No significant effect: safety stays normal and aggregate error "
            "(mean of tracking, TCP-tracking, force, and robot-current deviation terms) "
            f"increases by <={_fmt_pct(thresholds['no_effect_agg_increase'])}%."
        )
    return default_statement


def build_multiselect_options_and_answer(
    answer_format: Dict[str, Any],
    subseries: List[Dict[str, Any]],
    post_event_rows: List[Dict[str, Any]],
    mc_option_lookup: Dict[str, str],
    episode_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, str], str]:
    """
    Build exactly 4 multi-select options:
    1) Keep fixed options from template (if resolvable IDs).
    2) Sample random additional statements (excluding already selected IDs).
    3) Evaluate each option truth value.
    4) Shuffle and emit A-D options plus T/F answer string.
    """
    selected_ids: List[str] = []

    fixed_tokens = answer_format.get("fixed_options") or answer_format.get("fixed_statements") or []
    for token in fixed_tokens:
        option_id, _statement = resolve_fixed_option(token, mc_option_lookup)
        if option_id and option_id not in selected_ids:
            selected_ids.append(option_id)

    all_ids = sorted(mc_option_lookup.keys())
    remaining_ids = [opt_id for opt_id in all_ids if opt_id not in selected_ids]
    random.shuffle(remaining_ids)

    while len(selected_ids) < 4 and remaining_ids:
        selected_ids.append(remaining_ids.pop())

    selected_ids = selected_ids[:4]

    options_data: List[Tuple[str, str, Optional[bool]]] = []
    for opt_id in selected_ids:
        sampled_thresholds = sample_thresholds_for_statement(opt_id)
        statement = render_statement_with_thresholds(
            opt_id,
            mc_option_lookup.get(opt_id, opt_id),
            sampled_thresholds,
        )
        truth = evaluate_mc_statement(
            opt_id,
            subseries=subseries,
            post_event_rows=post_event_rows,
            thresholds=sampled_thresholds,
            episode_metadata=episode_metadata,
        )
        options_data.append((opt_id, statement, truth))

    random.shuffle(options_data)

    labels = ["A", "B", "C", "D"]
    options: Dict[str, str] = {}
    answer_chars: List[str] = []
    for idx, (_opt_id, statement, truth) in enumerate(options_data):
        if idx >= len(labels):
            break
        label = labels[idx]
        options[label] = statement
        answer_chars.append("T" if truth is True else "F")

    return options, "".join(answer_chars)


# ---------------------------------------------------------------------------
# Template filling  (level 2 specific)
# ---------------------------------------------------------------------------


def fill_template(
    template: Dict[str, Any],
    subseries: List[Dict[str, Any]],
    post_event_rows: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    mc_option_lookup: Dict[str, str],
    steps_ahead: Optional[int] = None,
    episode_metadata: Optional[Dict[str, Any]] = None,
    anomaly_description: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Fill a Level 2 question template.

    post_event_rows: rows from the episode after the subseries end
                     (used for ranking chunks).

    Returns dict with question/options/answer_format/event_id/answer, or None on failure.

    Template IDs:
      1 - signal_segment_ranking   : options = {A/B/C/D: encoded chunk}
    2 - intervention_outcome     : MC T/F with 4 shuffled statements
    3 - trajectory_outcome_multiselect: MC T/F with 4 shuffled statements
      4 - signal_value_prediction (numerical)
      5 - signal_value_prediction (tensor)
    """
    tid = template["id"]
    tmpl_text: str = template["template"]
    answer_format: Dict[str, Any] = template["answer_format"]
    onset_id = parse_event_id(post_event_rows[0].get("event", 0)) if post_event_rows else 0
    t_event = _first_timestamp_ms(post_event_rows)

    options: Dict[str, Any] = {}
    answer = None
    acceptance_bounds = None

    if tid == 1:
        chunks = sample_subsequent_chunks(post_event_rows, n_chunks=4, min_chunk=5, max_chunk=7)
        if len(chunks) < 4:
            return None

        ordered_chunks = list(chunks)
        random.shuffle(chunks)
        labels = ["A", "B", "C", "D"]
        options = {
            label: encode_chunk_without_timestamps(chunks[i])
            for i, label in enumerate(labels)
        }

        chunk_to_label = {id(chunks[i]): labels[i] for i in range(len(chunks))}
        answer = "".join(chunk_to_label[id(chunk)] for chunk in ordered_chunks)

        question = fill(tmpl_text, anomaly=anomaly_description)

    elif tid == 2:
        options, answer = build_multiselect_options_and_answer(
            answer_format=answer_format,
            subseries=subseries,
            post_event_rows=post_event_rows,
            mc_option_lookup=mc_option_lookup,
            episode_metadata=episode_metadata,
        )
        question = fill(tmpl_text, anomaly=anomaly_description)

    elif tid == 3:
        options, answer = build_multiselect_options_and_answer(
            answer_format=answer_format,
            subseries=subseries,
            post_event_rows=post_event_rows,
            mc_option_lookup=mc_option_lookup,
            episode_metadata=episode_metadata,
        )
        question = fill(tmpl_text, anomaly=anomaly_description)

    elif tid == 4:
        important_features = template.get("important_features")
        signal = pick_constrained_signal(subseries, important_features) if important_features else pick_scalar_signal(subseries)
        if signal is None:
            return None
        if steps_ahead is None or steps_ahead >= len(post_event_rows):
            return None
        target_row = post_event_rows[steps_ahead]
        n_ms = int(float(target_row.get("timestamp_ms", t_event))) - t_event
        signal_value = target_row.get(signal)
        if not isinstance(signal_value, (int, float, np.floating)):
            return None
        answer = round(float(signal_value), 6)
        question = fill(tmpl_text, anomaly=anomaly_description, signal=_signal_display_name(signal), n=n_ms)
        vals = [float(r[signal]) for r in subseries if isinstance(r.get(signal), (int, float, np.floating))]
        std = float(np.std(vals)) if vals else 0.0
        acceptance_bounds = {"signal": signal, "std": round(std, 6), "margin": round(std * 0.75, 6)}

    elif tid == 5:
        important_features = template.get("important_features")
        allowed_bases: Optional[set] = None
        if important_features:
            imp_set = set(important_features)
            allowed_bases = set()
            for feat in imp_set:
                m = re.match(r"^(.*)_(\d+)$", feat)
                if m and int(m.group(2)) in JOINT_INDEX_RANGE:
                    base = m.group(1)
                    if all(f"{base}_{i}" in imp_set for i in range(6)):
                        allowed_bases.add(base)
        joint_signal = pick_joint_indexed_signal_base(subseries, allowed_bases=allowed_bases or None)
        if joint_signal is None:
            return None
        if steps_ahead is None or steps_ahead >= len(post_event_rows):
            return None
        target_row = post_event_rows[steps_ahead]
        n_ms = int(float(target_row.get("timestamp_ms", t_event))) - t_event

        tensor_values: List[float] = []
        tensor_stds: List[float] = []
        for joint_idx in range(6):
            key = f"{joint_signal}_{joint_idx}"
            value = target_row.get(key)
            if not isinstance(value, (int, float, np.floating)):
                return None
            tensor_values.append(round(float(value), 6))
            vals = [float(r[key]) for r in subseries if isinstance(r.get(key), (int, float, np.floating))]
            tensor_stds.append(round(float(np.std(vals)) if vals else 0.0, 6))

        answer = "[" + ",".join(str(v) for v in tensor_values) + "]"
        _JOINT_SIGNAL_DISPLAY = {
            "setpoint_pos": "commanded joint positions",
            "setpoint_speed": "commanded joint velocities",
            "setpoint_acc": "commanded joint accelerations",
            "feedback_pos": "joint positions",
            "feedback_speed": "joint velocities",
            "effort_current": "joint motor currents",
            "effort_target_current": "target joint currents",
            "effort_target_torque": "joint motor torques",
            "control_output": "joint control outputs",
        }
        joint_signal_display = _JOINT_SIGNAL_DISPLAY.get(joint_signal, joint_signal.replace("_", " "))
        question = fill(tmpl_text, anomaly=anomaly_description, joint_signal=joint_signal_display, n=n_ms)
        acceptance_bounds = {"signal": joint_signal, "std": tensor_stds, "margin": [round(s * 0.75, 6) for s in tensor_stds]}

    else:
        logger.warning(f"Unknown template id: {tid}")
        return None

    return {
        "question": question,
        "answer_format": answer_format,
        "options": options,
        "event_id": onset_id,
        "answer": answer,
        "acceptance_bounds": acceptance_bounds,
    }


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def generate_level2_questions(
    datasets_dir: Path,
    output_dir: Path,
    templates: List[Dict[str, Any]],
    root_causes: Dict[int, Dict[str, Any]],
    events: List[Dict[str, Any]],
    n: int = 100,
    seed: Optional[int] = None,
    mc_option_lookup: Optional[Dict[str, str]] = None,
    datasets: Optional[List[str]] = None,
    relevance_specs: Optional[Dict[int, Dict[str, Any]]] = None,
    enumerate_mode: bool = False,
    uploader: Optional[HfStreamUploader] = None,
    completed_combos: Optional[set] = None,
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if mc_option_lookup is None:
        mc_option_lookup = {}

    relevance_specs = relevance_specs or {}

    # Load anomaly lookup for L1-style templates (7 = anomaly detection)
    anomaly_lookup = l1_load_anomaly_lookup(
        datasets_dir / "labelling" / "rca" / "anomalies.json"
    )

    # Comparative-template (t8) statements live as L1-only in mc_options.json
    # (usable_levels=[1]). Load them with level=1 explicitly, otherwise the
    # default L2 lookup misses mc_022/023/026 and the comparative builder
    # silently rejects every attempt.
    _l1_comparative_mc_lookup = load_mc_option_lookup(
        datasets_dir / "mc_options" / "mc_options.json", level=1
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    allowed = datasets if datasets else VALID_DATASETS
    by_dataset = discover_episodes_by_dataset(datasets_dir, allowed)
    if not by_dataset:
        raise FileNotFoundError(
            f"No normalized episode JSON files found under "
            f"{datasets_dir / 'normalized_episodes'} for datasets: {allowed}"
        )

    episodes_by_dataset = {ds: paths for ds, paths in by_dataset.items() if paths}
    available_datasets = list(episodes_by_dataset.keys())
    if not available_datasets:
        raise FileNotFoundError(
            f"No usable datasets with episodes found under {datasets_dir / 'normalized_episodes'}"
        )
    # LRU-bounded so enumerate-mode runs (which touch every episode) don't OOM.
    episode_cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    _EPISODE_CACHE_MAX = 50
    meta_cache: Dict[str, Dict[str, Any]] = {}

    def load_episode(path: Path) -> List[Dict[str, Any]]:
        key = str(path)
        if key in episode_cache:
            episode_cache.move_to_end(key)
            return episode_cache[key]
        raw = load_json(path)
        # Combined format {"baseline": [...], "counterfactual": [...]}
        if isinstance(raw, dict):
            raw = raw.get("counterfactual") or raw.get("baseline", [])
        episode_cache[key] = raw
        while len(episode_cache) > _EPISODE_CACHE_MAX:
            episode_cache.popitem(last=False)
        return episode_cache[key]

    def load_meta(ep_path: Path) -> Dict[str, Any]:
        key = str(ep_path)
        if key not in meta_cache:
            meta_path = ep_path.with_name(ep_path.stem + "_metadata.json")
            meta: Dict[str, Any] = {}
            if meta_path.exists():
                try:
                    meta = load_json(meta_path)
                except Exception:
                    pass
            meta_cache[key] = meta
        return meta_cache[key]

    # Pre-filter: only episodes containing anomalies. Metadata-driven only —
    # no per-row scan, which used to spike memory at scale.
    #   * If metadata carries ``fault_id`` (factorywave-style), keep iff
    #     ``fault_id != 0``.
    #   * If metadata lacks ``fault_id`` (aursad/vorausad-style), include the
    #     episode optimistically. Truly nominal episodes from these datasets
    #     get rejected at iteration time by the metadata-first
    #     ``pick_fault_label_from_meta_or_rows`` ``_fl == 0`` check, which is
    #     cheap because episode rows are loaded lazily through the LRU cache.
    def _ep_has_anomaly(p: Path) -> bool:
        meta = load_meta(p)
        fid = meta.get("fault_id")
        if fid is None:
            # No fault_id in metadata: trust the dataset's intent (these are
            # anomaly-detection corpora) and let iteration filter nominals.
            return True
        try:
            return int(float(fid)) != 0
        except (TypeError, ValueError):
            return True

    anomalous_episodes: List[Tuple[str, Path]] = []
    per_ds_counts: Dict[str, int] = {}
    for ds, paths in episodes_by_dataset.items():
        kept = 0
        for p in paths:
            if _ep_has_anomaly(p):
                anomalous_episodes.append((ds, p))
                kept += 1
        per_ds_counts[ds] = kept

    if not anomalous_episodes:
        logger.warning("No anomalous episodes found; cannot generate L2 questions.")
        return

    logger.info(
        "L2 eligible (anomalous-by-meta) episodes: "
        + ", ".join(f"{ds}={per_ds_counts.get(ds, 0)}" for ds in sorted(per_ds_counts))
        + f" — total {len(anomalous_episodes)}"
    )

    enum_iter = None
    if enumerate_mode:
        enum_combos = [
            (t, ds, ep)
            for t in templates
            for (ds, ep) in anomalous_episodes
        ]
        if completed_combos:
            before = len(enum_combos)
            enum_combos = [
                (t, ds, ep)
                for (t, ds, ep) in enum_combos
                if (int(t["id"]), ep.stem) not in completed_combos
            ]
            logger.info(f"[resume] skipping {before - len(enum_combos)} already-uploaded combos; {len(enum_combos)} remain")
        enum_iter = iter(enum_combos)
        max_total_attempts = len(enum_combos)
        logger.info(f"[enumerate] {len(enum_combos)} (template, episode) combos to attempt; -n={n} caps output")
    else:
        max_total_attempts = n * 20

    generated = 0
    attempts = 0

    while generated < n and attempts < max_total_attempts:
        attempts += 1

        if enum_iter is not None:
            try:
                template, ds, ep_path = next(enum_iter)
            except StopIteration:
                logger.info(f"[enumerate] all combos exhausted at {generated} questions")
                break
        else:
            template = random.choice(templates)
            ds, ep_path = random.choice(anomalous_episodes)

        steps_ahead = random.randint(*STEPS_AHEAD_RANGE)
        rows = load_episode(ep_path)
        if not isinstance(rows, list) or len(rows) < CONTEXT_MIN:
            continue

        _ep_meta = load_meta(ep_path)
        ep_fault_id = _ep_meta.get("fault_id")
        ep_task = _ep_meta.get("task", "")

        # Event-based templates need subseries positioned before an event;
        # L1-style templates just need any anomalous subseries.
        _EVENT_TEMPLATE_IDS = {1, 2, 3, 4, 5}
        subseries: List[Dict[str, Any]] = []
        post_event_rows: List[Dict[str, Any]] = []
        event_segment_rows: List[Dict[str, Any]] = []
        subseries_start_index = 0
        subseries_length = 0

        if template["id"] in _EVENT_TEMPLATE_IDS:
            sampled = sample_subseries_before_event(
                rows,
                CONTEXT_MIN,
                CONTEXT_MAX,
                min_post_event_after=MIN_POST_EVENT_TIMESTAMPS_AFTER,
                return_metadata=True,
            )
            subseries, post_event_rows, subseries_start_index, subseries_length = cast(
                Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int],
                sampled,
            )
            if not subseries:
                continue

            base_timestamp_ms = _first_timestamp_ms(subseries)
            subseries = normalize_timestamps(subseries, base_timestamp_ms)
            post_event_rows = normalize_timestamps(post_event_rows, base_timestamp_ms)

            event_segment_rows, _post_after_event_rows = split_event_segment(post_event_rows)
            if not event_segment_rows:
                continue

        # L1-style templates: handle inline on anomalous episodes.
        # The enumerated combo's episode is already drawn from anomalous_episodes
        # (built by load_meta(p).get("fault_id") earlier), so we use it directly
        # instead of re-picking. In random mode we restrict the pool to
        # anomalous_episodes too, so every L2 question is grounded in an actual
        # anomaly — never a nominal episode.
        if template["id"] in _L1_STYLE_TEMPLATE_IDS:
            if enum_iter is not None:
                _ds, _ep_path = ds, ep_path
            else:
                _ds, _ep_path = random.choice(anomalous_episodes)
            _raw = load_json(_ep_path)
            if isinstance(_raw, dict):
                _raw = _raw.get("baseline", _raw.get("flat", []))
            if not isinstance(_raw, list) or len(_raw) < CONTEXT_MIN:
                continue
            # Fault: prefer metadata's fault_id; fall back to most-common
            # non-zero fault_label. This recovers sparse-anomaly episodes
            # (most rows nominal but the episode has a short anomaly burst)
            # that pick_fault_label wrongly picks as 0.
            _fl = pick_fault_label_from_meta_or_rows(_raw, load_meta(_ep_path))
            if _fl == 0:
                continue
            _rc = root_causes.get(_fl, {})
            _anomaly = _anomaly_inline_name(_fl, _rc)
            _machine_id = DATASET_MACHINE_ID.get(_ds, -1)

            _meta_path = _ep_path.with_name(_ep_path.stem + "_metadata.json")
            _ep_task = ""
            if _meta_path.exists():
                try:
                    _meta = load_json(_meta_path)
                    _ep_task = _meta.get("task", "")
                except Exception:
                    pass

            from src.question_generation.utils.relevance import (
                sample_with_relevance as _sample_with_relevance,
                validate_relevance as _validate_relevance,
                relevance_report as _relevance_report,
            )
            _spec = relevance_specs.get(_fl) if relevance_specs else None
            _sampled = _sample_with_relevance(_raw, _fl, _spec, _ep_task, CONTEXT_MIN, CONTEXT_MAX)
            if _sampled is None:
                continue
            _sub, _start, _sampler_tag = _sampled
            _sub = normalize_timestamps(_sub, _first_timestamp_ms(_sub))
            if not _validate_relevance(_sub, _spec, _ep_task):
                continue

            _tmpl = template["template"]
            _af = template["answer_format"]
            _tid = template["id"]
            _opts: Dict[str, Any] = {}
            _ans: Any = None
            _bounds = None

            if _tid == 6:
                # Sample up to 3 distinct phases (excluding first/last when ≥3
                # phases exist; otherwise use all phases) so each (template,
                # episode, subseries) combo yields multiple phase-isolation
                # questions. Each is written and uploaded independently; the
                # common write path is skipped via the `continue` at the end.
                _phase_pool = l1_pick_phase_isolation_candidates(_sub, n=3)
                if not _phase_pool:
                    continue
                _imp_local = template.get("important_features")
                if _imp_local:
                    _keep_local = set(_imp_local) | {"timestamp_ms"}
                    _ctx_rows_local = [{k: v for k, v in r.items() if k in _keep_local} for r in _sub]
                else:
                    _ctx_rows_local = _sub
                _context_local = build_context(_ctx_rows_local)
                _wrote_any = False
                for _ph in _phase_pool:
                    if generated >= n:
                        break
                    _pn, _pi, _pl = _ph
                    _gt_t = int(round(float(_sub[_pi]["timestamp_ms"])))
                    _low_idx = max(0, _pi - 3)
                    _high_idx = min(len(_sub) - 1, _pi + 3)
                    _t_low = int(round(float(_sub[_low_idx]["timestamp_ms"])))
                    _t_high = int(round(float(_sub[_high_idx]["timestamp_ms"])))
                    _ans_local = _gt_t
                    _bounds_local = {"min": _t_low, "max": _t_high}
                    _tmpl_filled_local = fill(
                        _tmpl, anomaly=_anomaly,
                        phase=_phase_display_name(_pn, _ep_task),
                        window_length=_pl + 5,
                    )
                    item = {
                        "id": str(uuid.uuid4()),
                        "level": 2,
                        "template_id": _tid,
                        "template_type": template["type"],
                        "hides": template.get("hides", []),
                        "question": _tmpl_filled_local,
                        "options": {},
                        "answer": _ans_local,
                        "acceptance_bounds": _bounds_local,
                        "provenance": {
                            "dataset": _ds, "episode": _ep_path.stem,
                            "fault_label": _fl,
                            "subseries_start_index": _start,
                            "subseries_length": len(_sub),
                            "phase_name": _pn,
                            "phase_start_in_subseries": _pi,
                            "phase_length": _pl,
                            "relevance": _relevance_report(_sub, _fl, _spec, _ep_task, _sampler_tag),
                        },
                        "context": _context_local,
                    }
                    out_path = output_dir / f"level2_{generated:04d}.json"
                    with out_path.open("w", encoding="utf-8") as f:
                        json.dump(item, f, indent=2)
                    logger.info(f"✓ [{generated + 1}/{n}] {out_path.name} (template 6, {_ds}, phase={_pn})")
                    generated += 1
                    if uploader is not None:
                        uploader.maybe_flush(generated)
                    _wrote_any = True
                if not _wrote_any:
                    continue
                continue  # skip the common write path; t6 wrote its own items

            elif _tid == 7:
                _opts, _ans = l1_build_anomaly_single_select(_fl, root_causes, anomaly_lookup)
                _tmpl_filled = _tmpl  # no {anomaly} placeholder — the model must identify it

            elif _tid == 10:
                _rn = {0: "Universal Robots UR3e", 2: "Agile Robots Yu 5 Industrial", 3: "KUKA KR 10 R1100-2"}
                if _machine_id not in _rn:
                    continue
                _correct = _rn[_machine_id]
                _fixed = _af.get("fixed_options", list(_rn.values()))
                _labels = ["A", "B", "C"][:len(_fixed)]
                _entries = list(_fixed)
                random.shuffle(_entries)
                _opts = {}
                _ans = "A"
                for _i, _name in enumerate(_entries):
                    _opts[_labels[_i]] = _name
                    if _name == _correct:
                        _ans = _labels[_i]
                _tmpl_filled = fill(_tmpl, anomaly=_anomaly)

            elif _tid == 8:
                # Comparative multi-select on TWO anomalous episodes (mirror of
                # L1 t3 but both streams have anomalies). Enumerated ep is one
                # stream; the second is sampled at random from anomalous_episodes.
                if len(anomalous_episodes) < 2:
                    continue
                _other_pool = [(d, p) for (d, p) in anomalous_episodes if p != _ep_path]
                if not _other_pool:
                    continue
                _ds_b, _ep_path_b = random.choice(_other_pool)
                _raw_b = load_json(_ep_path_b)
                if isinstance(_raw_b, dict):
                    _raw_b = _raw_b.get("baseline", _raw_b.get("flat", []))
                if not isinstance(_raw_b, list) or len(_raw_b) < CONTEXT_MIN:
                    continue
                _fl_b = pick_fault_label_from_meta_or_rows(_raw_b, load_meta(_ep_path_b))
                _meta_b_path = _ep_path_b.with_name(_ep_path_b.stem + "_metadata.json")
                _ep_task_b = ""
                if _meta_b_path.exists():
                    try:
                        _ep_task_b = load_json(_meta_b_path).get("task", "")
                    except Exception:
                        pass
                _spec_b = relevance_specs.get(_fl_b) if relevance_specs else None
                _sampled_b = _sample_with_relevance(
                    _raw_b, _fl_b, _spec_b, _ep_task_b, CONTEXT_MIN, CONTEXT_MAX
                )
                if _sampled_b is None:
                    continue
                _sub_b, _start_b, _sampler_b = _sampled_b
                _sub_b = normalize_timestamps(_sub_b, _first_timestamp_ms(_sub_b))
                if not _validate_relevance(_sub_b, _spec_b, _ep_task_b):
                    continue
                _machine_id_b = DATASET_MACHINE_ID.get(_ds_b, -1)
                _result = l1_build_comparative_multi_select(
                    fault_a=_fl, fault_b=_fl_b,
                    machine_id_a=_machine_id, machine_id_b=_machine_id_b,
                    task_id_a=_ep_task, task_id_b=_ep_task_b,
                    rows_a=_sub, rows_b=_sub_b,
                    mc_lookup=_l1_comparative_mc_lookup,
                )
                if _result is None:
                    continue
                _opts, _ans = _result
                _tmpl_filled = _tmpl  # comparative prompt has no placeholders

                # Build dual-stream context and override the default emit path.
                _imp = template.get("important_features")
                if _imp:
                    _keep = set(_imp) | {"timestamp_ms"}
                    _ctx_a = [{k: v for k, v in r.items() if k in _keep} for r in _sub]
                    _ctx_b = [{k: v for k, v in r.items() if k in _keep} for r in _sub_b]
                else:
                    _ctx_a, _ctx_b = _sub, _sub_b
                _context_dual = {
                    "series_a": build_context(_ctx_a),
                    "series_b": build_context(_ctx_b),
                }
                item = {
                    "id": str(uuid.uuid4()),
                    "level": 2,
                    "template_id": _tid,
                    "template_type": template["type"],
                    "hides": template.get("hides", []),
                    "question": _tmpl_filled,
                    "options": _opts,
                    "answer": _ans,
                    "acceptance_bounds": None,
                    "provenance": {
                        "dataset_a": _ds, "episode_a": _ep_path.stem,
                        "fault_label_a": _fl, "machine_id_a": _machine_id,
                        "dataset_b": _ds_b, "episode_b": _ep_path_b.stem,
                        "fault_label_b": _fl_b, "machine_id_b": _machine_id_b,
                    },
                    "context": _context_dual,
                }
                out_path = output_dir / f"level2_{generated:04d}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(item, f, indent=2)
                logger.info(f"✓ [{generated + 1}/{n}] {out_path.name} (template 8, {_ds}+{_ds_b})")
                generated += 1
                if uploader is not None:
                    uploader.maybe_flush(generated)
                continue

            elif _tid == 9:
                # Severity ranking on 4 anomalous segments (mirror of L1 t5).
                if len(anomalous_episodes) < 4:
                    continue
                _other_pool = [(d, p) for (d, p) in anomalous_episodes if p != _ep_path]
                if len(_other_pool) < 3:
                    continue
                _sampled_eps = [(_ds, _ep_path)] + random.sample(_other_pool, 3)
                _segments: List[Tuple[List[Dict[str, Any]], int]] = []
                _seg_tasks: List[str] = []
                _bad = False
                for (_sd, _sp) in _sampled_eps:
                    _srows = load_json(_sp)
                    if isinstance(_srows, dict):
                        _srows = _srows.get("baseline", _srows.get("flat", []))
                    if not isinstance(_srows, list) or len(_srows) < 5:
                        _bad = True
                        break
                    _smeta_dict: Dict[str, Any] = {}
                    _smeta = _sp.with_name(_sp.stem + "_metadata.json")
                    _stask = ""
                    if _smeta.exists():
                        try:
                            _smeta_dict = load_json(_smeta) or {}
                            _stask = _smeta_dict.get("task", "")
                        except Exception:
                            pass
                    _sfl = pick_fault_label_from_meta_or_rows(_srows, _smeta_dict)
                    _segments.append((_srows, _sfl))
                    _seg_tasks.append(_stask)
                if _bad or len(_segments) < 4:
                    continue
                _result = l1_build_severity_ranking(
                    _segments, root_causes,
                    important_features=template.get("important_features"),
                    relevance_specs=relevance_specs,
                    tasks=_seg_tasks,
                )
                if _result is None:
                    continue
                _opts, _ans = _result
                _tmpl_filled = _tmpl  # ranking prompt has no placeholders
                item = {
                    "id": str(uuid.uuid4()),
                    "level": 2,
                    "template_id": _tid,
                    "template_type": template["type"],
                    "hides": template.get("hides", []),
                    "question": _tmpl_filled,
                    "options": _opts,
                    "answer": _ans,
                    "acceptance_bounds": None,
                    "provenance": {
                        "episodes": [
                            {"dataset": d, "episode": p.stem}
                            for (d, p) in _sampled_eps
                        ],
                    },
                    "context": {},
                }
                out_path = output_dir / f"level2_{generated:04d}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(item, f, indent=2)
                logger.info(f"✓ [{generated + 1}/{n}] {out_path.name} (template 9, severity-rank)")
                generated += 1
                if uploader is not None:
                    uploader.maybe_flush(generated)
                continue

            else:
                # No remaining unimplemented L1-style template ids.
                continue

            _imp = template.get("important_features")
            if _imp:
                _keep = set(_imp) | {"timestamp_ms"}
                _ctx_rows = [{k: v for k, v in r.items() if k in _keep} for r in _sub]
            else:
                _ctx_rows = _sub

            item = {
                "id": str(uuid.uuid4()),
                "level": 2,
                "template_id": _tid,
                "template_type": template["type"],
            "hides": template.get("hides", []),
                "question": _tmpl_filled,
                "options": _opts,
                "answer": _ans,
                "acceptance_bounds": _bounds,
                "provenance": {"dataset": _ds, "episode": _ep_path.stem, "fault_label": _fl, "subseries_start_index": _start, "subseries_length": len(_sub), "relevance": _relevance_report(_sub, _fl, _spec, _ep_task, _sampler_tag)},
                "context": build_context(_ctx_rows),
            }

            out_path = output_dir / f"level2_{generated:04d}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(item, f, indent=2)
            logger.info(f"✓ [{generated + 1}/{n}] {out_path.name} (template {_tid}, {_ds})")
            generated += 1
            if uploader is not None:
                uploader.maybe_flush(generated)
            continue

        if ds == "simulations":
            effective_mc_lookup = {k: v for k, v in mc_option_lookup.items() if k not in SIMULATION_EXCLUDED_MC_IDS}
        else:
            effective_mc_lookup = {k: v for k, v in mc_option_lookup.items() if k not in NON_SIMULATION_EXCLUDED_MC_IDS}

        if template.get("type") == "predictive":
            effective_mc_lookup = {k: v for k, v in effective_mc_lookup.items() if k not in PREDICTIVE_EXCLUDED_MC_IDS}

        ep_metadata: Optional[Dict[str, Any]] = None
        if ds == "simulations":
            meta_path = ep_path.parent / f"{ep_path.stem}_metadata.json"
            if meta_path.exists():
                try:
                    with meta_path.open(encoding="utf-8") as _f:
                        raw_meta = json.load(_f)
                    ep_metadata = raw_meta.get("counterfactual", {})
                except Exception:
                    pass

        # Get anomaly description from metadata fault_id
        _rc = root_causes.get(ep_fault_id, {}) if ep_fault_id else {}
        _anomaly_desc = _anomaly_inline_name(ep_fault_id, _rc) if ep_fault_id else ""

        filled = fill_template(
            template, subseries, post_event_rows, events, effective_mc_lookup,
            steps_ahead=steps_ahead, episode_metadata=ep_metadata,
            anomaly_description=_anomaly_desc,
        )
        if filled is None:
            continue

        subseries_with_event = subseries + event_segment_rows
        important_features = template.get("important_features")
        if important_features:
            keep = set(important_features) | {"timestamp_ms"}
            subseries_with_event = [{k: v for k, v in row.items() if k in keep} for row in subseries_with_event]
        context = build_context(subseries_with_event)

        item = {
            "id": str(uuid.uuid4()),
            "level": 2,
            "template_id": template["id"],
            "template_type": template["type"],
            "hides": template.get("hides", []),
            "question": filled["question"],
            "options": filled["options"],
            "answer": filled["answer"],
            "acceptance_bounds": filled.get("acceptance_bounds"),
            "provenance": {
                "dataset": ds,
                "episode": ep_path.stem,
                "subseries_start_index": subseries_start_index,
                "subseries_length": len(subseries_with_event),
            },
            "context": context,
        }

        out_path = output_dir / f"level2_{generated:04d}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)

        logger.info(
            f"✓ [{generated + 1}/{n}] {out_path.name} "
            f"(template {template['id']}, {ds})"
        )
        generated += 1
        if uploader is not None:
            uploader.maybe_flush(generated)

    if uploader is not None:
        uploader.flush_remaining()

    if generated < n:
        logger.warning(f"Only generated {generated}/{n} questions after {attempts} attempts")
    else:
        logger.info(f"Done: {generated} questions written to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Level 2 (Intervention Reasoning) Q&A pairs."
    )
    repo_root = Path(__file__).resolve().parents[3]

    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=repo_root / "data",
        help="Root data directory (default: <repo>/data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "output" / "questions" / "level2",
        help="Output directory (default: <repo>/output/questions/level2)",
    )
    parser.add_argument("-n", type=int, default=100, help="Number of questions to generate (cap; in --enumerate mode this is an upper bound, not a target)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--enumerate",
        dest="enumerate_mode",
        action="store_true",
        help="Walk every (template x episode) combination deterministically instead "
             "of random sampling. -n becomes an upper cap. Combinations whose "
             "episode does not satisfy the template's preconditions are skipped.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Datasets to sample from (default: all). Choices: {VALID_DATASETS}",
    )
    add_streaming_args(parser)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    templates = load_templates(Path(__file__).with_name("question_template.json"))
    root_causes = load_root_causes(args.datasets_dir / "labelling" / "rca" / "root_causes.json")
    events = load_events(args.datasets_dir / "labelling" / "events.json")
    mc_option_lookup = load_mc_option_lookup(
        args.datasets_dir / "mc_options" / "mc_options.json",
        level=2,
    )

    from src.question_generation.utils.relevance import (
        is_enabled as _relevance_enabled,
        load_specs as _load_relevance_specs,
    )
    relevance_specs = (
        _load_relevance_specs(args.datasets_dir / "labelling" / "rca" / "relevance_specs.json")
        if _relevance_enabled() else {}
    )

    args.output.mkdir(parents=True, exist_ok=True)
    uploader = make_uploader_from_args(args, level=2, output_dir=args.output)
    completed_combos = None
    if getattr(args, "resume_from_hf", False) and args.hf_dataset_folder:
        completed_combos, max_shard_idx = list_completed_combos(
            repo_id=args.hf_repo,
            dataset_folder=args.hf_dataset_folder,
            level=2,
        )
        # Resume from the next shard index so we don't overwrite existing files.
        if uploader is not None and max_shard_idx > 0:
            uploader.batch_index = max_shard_idx

    generate_level2_questions(
        datasets_dir=args.datasets_dir,
        output_dir=args.output,
        templates=templates,
        root_causes=root_causes,
        events=events,
        mc_option_lookup=mc_option_lookup,
        n=args.n,
        seed=args.seed,
        datasets=args.datasets,
        relevance_specs=relevance_specs,
        enumerate_mode=args.enumerate_mode,
        uploader=uploader,
        completed_combos=completed_combos,
    )


if __name__ == "__main__":
    main()
