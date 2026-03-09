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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from src.question_generation.utils.io import load_events, load_json, load_root_causes, load_templates
from src.question_generation.utils.template import (
    build_context,
    discover_episodes_by_dataset,
    encode_chunk,
    fill,
    fill_event_description,
    get_last_timestamp,
    pick_scalar_signal,
)
from src.question_generation.utils.time_series import (
    parse_event_id,
    pick_fault_label,
    sample_subseries_before_event,
)
from src.question_generation.level2.mc_truth import DEFAULT_THRESHOLDS, evaluate_mc_statement

logger = logging.getLogger(__name__)

VALID_DATASETS = ["inter_aursad", "inter_vorausad"]

DIFFICULTY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy":   {"steps_ahead_range": (1, 2),  "context_min": 65, "context_max": 90},
    "medium": {"steps_ahead_range": (3, 5),  "context_min": 32, "context_max": 64},
    "hard":   {"steps_ahead_range": (6, 10), "context_min": 16, "context_max": 31},
}
DIFFICULTIES = list(DIFFICULTY_CONFIGS.keys())



EXCLUDED_JOINT_SIGNALS = {"joint_voltage", "joint_temp", "joint_mode"}
JOINT_INDEX_RANGE = set(range(6))
MIN_POST_EVENT_TIMESTAMPS_AFTER = 5


def pick_joint_indexed_signal_base(subseries: List[Dict[str, Any]]) -> Optional[str]:
    """
    Pick a base signal name that has indexed variants for all joints 0..5.

    Example valid base: "setpoint_pos" (requires setpoint_pos_0 ... setpoint_pos_5).
    Excludes: joint_voltage, joint_temp, joint_mode.
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
    difficulty: str = "medium",
    steps_ahead: Optional[int] = None,
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
    t = get_last_timestamp(subseries)
    t_event = _first_timestamp_ms(post_event_rows)

    onset_id = parse_event_id(post_event_rows[0].get("event", 0)) if post_event_rows else 0
    event_obj = next((e for e in events if e["id"] == onset_id), random.choice(events))
    event_desc = fill_event_description(event_obj, subseries, t, post_event_rows)

    options: Dict[str, Any] = {}
    answer = None

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

        question = fill(tmpl_text, t_event=t_event, event=event_desc)

    elif tid == 2:
        options, answer = build_multiselect_options_and_answer(
            answer_format=answer_format,
            subseries=subseries,
            post_event_rows=post_event_rows,
            mc_option_lookup=mc_option_lookup,
        )
        question = fill(tmpl_text, event=event_desc, t_event=t_event)

    elif tid == 3:
        options, answer = build_multiselect_options_and_answer(
            answer_format=answer_format,
            subseries=subseries,
            post_event_rows=post_event_rows,
            mc_option_lookup=mc_option_lookup,
        )
        question = fill(tmpl_text, event=event_desc, t_event=t_event)

    elif tid == 4:
        signal = pick_scalar_signal(subseries)
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
        question = fill(tmpl_text, event=event_desc, t_event=t_event, signal=signal, n=n_ms)

    elif tid == 5:
        joint_signal = pick_joint_indexed_signal_base(subseries)
        if joint_signal is None:
            return None
        if steps_ahead is None or steps_ahead >= len(post_event_rows):
            return None
        target_row = post_event_rows[steps_ahead]
        n_ms = int(float(target_row.get("timestamp_ms", t_event))) - t_event

        tensor_values: List[float] = []
        for joint_idx in range(6):
            key = f"{joint_signal}_{joint_idx}"
            value = target_row.get(key)
            if not isinstance(value, (int, float, np.floating)):
                return None
            tensor_values.append(round(float(value), 6))

        answer = "_".join(str(v) for v in tensor_values)
        question = fill(tmpl_text, event=event_desc, t_event=t_event, joint_signal=joint_signal, n=n_ms)

    else:
        logger.warning(f"Unknown template id: {tid}")
        return None

    return {
        "question": question,
        "answer_format": answer_format,
        "options": options,
        "event_id": event_obj["id"],
        "answer": answer,
        "difficulty": difficulty,
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
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if mc_option_lookup is None:
        mc_option_lookup = {}

    output_dir.mkdir(parents=True, exist_ok=True)

    by_dataset = discover_episodes_by_dataset(datasets_dir, VALID_DATASETS)
    if not by_dataset:
        raise FileNotFoundError(
            f"No normalized episode JSON files found under "
            f"{datasets_dir / 'normalized_episodes'} for datasets: {VALID_DATASETS}"
        )

    episodes_by_dataset = {ds: paths for ds, paths in by_dataset.items() if paths}
    available_datasets = list(episodes_by_dataset.keys())
    if not available_datasets:
        raise FileNotFoundError(
            f"No usable datasets with episodes found under {datasets_dir / 'normalized_episodes'}"
        )
    episode_cache: Dict[str, List[Dict[str, Any]]] = {}

    def load_episode(path: Path) -> List[Dict[str, Any]]:
        key = str(path)
        if key not in episode_cache:
            episode_cache[key] = load_json(path)
        return episode_cache[key]

    generated = 0
    attempts = 0
    max_total_attempts = n * 20

    while generated < n and attempts < max_total_attempts:
        attempts += 1

        difficulty = random.choice(DIFFICULTIES)
        diff_cfg = DIFFICULTY_CONFIGS[difficulty]
        context_min = diff_cfg["context_min"]
        context_max = diff_cfg["context_max"]
        steps_ahead = random.randint(*diff_cfg["steps_ahead_range"])

        ds = random.choice(available_datasets)
        ep_path = random.choice(episodes_by_dataset[ds])
        rows = load_episode(ep_path)
        if not isinstance(rows, list) or len(rows) < context_min:
            continue

        sampled = sample_subseries_before_event(
            rows,
            context_min,
            context_max,
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

        event_segment_rows, post_after_event_rows = split_event_segment(post_event_rows)
        if not event_segment_rows:
            continue
        if len(post_after_event_rows) < MIN_POST_EVENT_TIMESTAMPS_AFTER:
            continue

        template = random.choice(templates)

        filled = fill_template(
            template, subseries, post_event_rows, events, mc_option_lookup,
            difficulty=difficulty, steps_ahead=steps_ahead,
        )
        if filled is None:
            continue

        subseries_with_event = subseries + event_segment_rows
        context = build_context(subseries_with_event)

        item = {
            "id": str(uuid.uuid4()),
            "level": 2,
            "difficulty": difficulty,
            "template_id": template["id"],
            "template_type": template["type"],
            "question": filled["question"],
            "options": filled["options"],
            "answer": filled["answer"],
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
    parser.add_argument("-n", type=int, default=100, help="Number of questions to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
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

    generate_level2_questions(
        datasets_dir=args.datasets_dir,
        output_dir=args.output,
        templates=templates,
        root_causes=root_causes,
        events=events,
        mc_option_lookup=mc_option_lookup,
        n=args.n,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
