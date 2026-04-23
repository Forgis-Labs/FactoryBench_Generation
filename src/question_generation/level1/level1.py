"""
Level 1 question generator: State Understanding.

Reads normalized episode JSON files from aursad or vorausad datasets,
samples random sub-series, and fills Level 1 question templates.
Ground truth is derived directly from episode labels (fault_label) and metadata.

Output: output/questions/level1/level1_{NNNN}.json

Usage:
    python -m src.question_generation.level1.level1 -n 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data._decimation import decimate_dataframe
from src.question_generation.utils.io import load_json, load_root_causes, load_templates
from src.question_generation.utils.template import (
    build_context,
    discover_episodes_by_dataset,
    encode_chunk,
    fill,
)
from src.question_generation.utils.time_series import (
    pick_fault_label,
)

logger = logging.getLogger(__name__)

VALID_DATASETS = ["inter_aursad", "inter_vorausad", "factorywave"]
SEVERITY_ORDER: Dict[str, int] = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

# Precise per-fault ranking loaded from anomaly_ranking.json (fault_id → rank 1..N, higher = more severe)
_ANOMALY_RANKING: Dict[int, int] = {}

# Maps dataset name → machine_id as defined in machines.json
DATASET_MACHINE_ID: Dict[str, int] = {
    "inter_aursad": 0,    # UR3e
    "inter_vorausad": 2,  # Yu cobot
    "factorywave": 0,     # UR3e (FactoryWave real-robot recordings)
}

_NO_ANOMALY_DESC = "No anomaly is present; the machine is operating nominally."

CONTEXT_MIN = 16
CONTEXT_MAX = 90

# Phase index → human-readable name per task
PHASE_NAMES: Dict[str, Dict[str, str]] = {
    "pick_and_place": {
        "0": "approach above the pick position",
        "1": "descent toward the object",
        "2": "settling pause before grasping",
        "3": "gripper closing phase",
        "4": "lifting phase",
        "5": "lateral transfer toward the bin",
        "6": "lowering into the bin",
        "7": "gripper opening phase",
        "8": "retraction away from the bin",
        "9": "return to home position",
    },
    "screwing": {
        "0": "approach above the fastener",
        "1": "descent to engage the fastener",
        "2": "screwing phase",
        "3": "disengagement from the fastener",
        "4": "retraction to safe height",
        "5": "descent to re-engage the fastener",
        "6": "loosening phase",
        "7": "gripper engagement phase",
        "8": "retraction to home position",
    },
    "peg_in_hole": {
        "0": "alignment above the hole",
        "1": "insertion into the hole",
        "2": "disengagement from the peg",
        "3": "rise back above the hole",
        "4": "retraction to home position",
        "5": "alignment above the object",
        "6": "gripper engagement phase",
        "7": "rise back above the object",
        "8": "retraction to home position",
    },
}

# Variable name → human-readable signal name
SIGNAL_DISPLAY_NAMES: Dict[str, str] = {}
# Build automatically for indexed signals
for _i in range(6):
    SIGNAL_DISPLAY_NAMES[f"feedback_pos_{_i}"] = f"the position of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"feedback_speed_{_i}"] = f"the velocity of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"setpoint_pos_{_i}"] = f"the commanded position of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"setpoint_speed_{_i}"] = f"the commanded velocity of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"setpoint_acc_{_i}"] = f"the commanded acceleration of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"effort_current_{_i}"] = f"the motor current of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"effort_target_current_{_i}"] = f"the target current of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"effort_target_torque_{_i}"] = f"the motor torque of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"control_output_{_i}"] = f"the control output of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"joint_temp_{_i}"] = f"the temperature of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"joint_mode_{_i}"] = f"the mode of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"joint_voltage_{_i}"] = f"the voltage of joint {_i}"
for _i, _axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
    SIGNAL_DISPLAY_NAMES[f"setpoint_tcp_{_i}"] = f"the commanded TCP {_axis}"
    SIGNAL_DISPLAY_NAMES[f"setpoint_tcp_speed_{_i}"] = f"the commanded TCP speed {_axis}"
    SIGNAL_DISPLAY_NAMES[f"feedback_tcp_{_i}"] = f"the TCP {_axis}"
    SIGNAL_DISPLAY_NAMES[f"feedback_tcp_speed_{_i}"] = f"the TCP speed {_axis}"
for _i, _axis in enumerate(["x", "y", "z"]):
    SIGNAL_DISPLAY_NAMES[f"true_force_{_i}"] = f"the TCP force {_axis}"
    SIGNAL_DISPLAY_NAMES[f"true_force_{_i + 3}"] = f"the TCP torque {_axis}"
    SIGNAL_DISPLAY_NAMES[f"est_contact_force_{_i}"] = f"the estimated contact force {_axis}"
    SIGNAL_DISPLAY_NAMES[f"vibration_{_i}"] = f"the vibration {_axis}"
SIGNAL_DISPLAY_NAMES["gripper_command"] = "the gripper force"
SIGNAL_DISPLAY_NAMES["robot_mode"] = "the robot mode"
SIGNAL_DISPLAY_NAMES["safety_mode"] = "the safety mode"
SIGNAL_DISPLAY_NAMES["runtime_state"] = "the runtime state"
SIGNAL_DISPLAY_NAMES["speed_scaling"] = "the speed scaling"
SIGNAL_DISPLAY_NAMES["robot_current"] = "the robot current"
SIGNAL_DISPLAY_NAMES["main_voltage"] = "the main voltage"
SIGNAL_DISPLAY_NAMES["robot_voltage"] = "the robot voltage"
SIGNAL_DISPLAY_NAMES["tool_momentum"] = "the tool momentum"


def _phase_display_name(phase_raw: str, task: str = "pick_and_place") -> str:
    """Convert a phase index/name to a human-readable name."""
    task_phases = PHASE_NAMES.get(task, PHASE_NAMES.get("pick_and_place", {}))
    name = task_phases.get(str(phase_raw))
    if name:
        return name
    # Already a name string (not an index)
    return str(phase_raw).replace("_", " ")


def _signal_display_name(signal: str) -> str:
    """Convert a signal variable name to a human-readable name."""
    if signal in SIGNAL_DISPLAY_NAMES:
        return SIGNAL_DISPLAY_NAMES[signal]
    return signal.replace("_", " ")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _first_timestamp_ms(rows: List[Dict[str, Any]]) -> int:
    for row in rows:
        ts = row.get("timestamp_ms")
        try:
            if ts is not None:
                return int(float(ts))
        except (TypeError, ValueError):
            continue
    return 0


def normalize_timestamps(rows: List[Dict[str, Any]], base: int) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        r = dict(row)
        ts = r.get("timestamp_ms")
        try:
            if ts is not None:
                r["timestamp_ms"] = int(float(ts)) - base
        except (TypeError, ValueError):
            pass
        out.append(r)
    return out


def sample_subseries(
    rows: List[Dict[str, Any]],
    min_len: int = CONTEXT_MIN,
    max_len: int = CONTEXT_MAX,
) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """Return (subseries, start_index) or None if the episode is too short."""
    n = len(rows)
    if n < min_len:
        return None
    length = random.randint(min_len, min(max_len, n))
    start = random.randint(0, n - length)
    return rows[start : start + length], start


def load_anomaly_ranking(path: Path) -> Dict[int, int]:
    """Load anomaly_ranking.json as {fault_id: rank}.

    Higher rank = more severe (the JSON lists least-to-most, so rank 1 is
    the least severe and rank N is the most severe).
    """
    try:
        raw = load_json(path)
    except Exception as exc:
        logger.warning(f"Could not load anomaly ranking from {path}: {exc}")
        return {}
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("ranking_least_to_most_severe", [])
    return {
        int(entry["fault_id"]): int(entry["rank"])
        for entry in entries
        if isinstance(entry, dict) and "fault_id" in entry and "rank" in entry
    }


def get_severity_rank(fault_id: int, root_cause: Dict[str, Any]) -> int:
    """Return a numeric severity rank for a fault.

    Uses the precise per-fault ranking from anomaly_ranking.json if available,
    otherwise falls back to the coarse severity_levels mapping.
    """
    if _ANOMALY_RANKING and fault_id in _ANOMALY_RANKING:
        return _ANOMALY_RANKING[fault_id]
    levels = root_cause.get("severity_levels", ["none"])
    return max((SEVERITY_ORDER.get(s, 0) for s in levels), default=0)


def normalize_mc_option_id(value: str) -> str:
    text = str(value).strip()
    match = re.match(r"^(?:mc|l2_mc)_(\d+)$", text, re.IGNORECASE)
    if not match:
        return text
    return f"mc_{int(match.group(1)):03d}"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_mc_option_lookup(path: Path, level: int) -> Dict[str, str]:
    """Load MC options as a map: id -> statement, filtered by level."""
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


def load_anomaly_lookup(path: Path) -> Dict[str, str]:
    """Load anomalies.json as {anomaly_name: description}."""
    try:
        raw = load_json(path)
    except Exception as exc:
        logger.warning(f"Could not load anomalies from {path}: {exc}")
        return {}
    if not isinstance(raw, list):
        return {}
    return {
        item["anomaly_name"]: item["description"]
        for item in raw
        if isinstance(item, dict) and "anomaly_name" in item
    }


def load_dataset_index(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load dataset.json as {dataset_id: dataset_dict}."""
    try:
        raw = load_json(path)
    except Exception as exc:
        logger.warning(f"Could not load dataset index from {path}: {exc}")
        return {}
    if not isinstance(raw, list):
        return {}
    return {
        item["dataset_id"]: item
        for item in raw
        if isinstance(item, dict) and "dataset_id" in item
    }


# ---------------------------------------------------------------------------
# Answer builders (one per template)
# ---------------------------------------------------------------------------


def build_anomaly_single_select(
    fault_label: int,
    root_causes: Dict[int, Dict[str, Any]],
    anomaly_lookup: Dict[str, str],
) -> Tuple[Dict[str, str], str]:
    """
    Build 4 single-select options + single letter answer for template 2.
    One option is the correct anomaly (or 'no anomaly' for normal episodes),
    the rest are distractors.
    """
    root_cause = root_causes.get(fault_label, root_causes.get(0, {}))
    true_anomalies: List[str] = root_cause.get("possible_anomalies", [])
    is_normal = fault_label == 0 or not true_anomalies

    distractor_pool = [a for a in anomaly_lookup if a not in true_anomalies]
    random.shuffle(distractor_pool)

    if is_normal:
        correct_desc = _NO_ANOMALY_DESC
    else:
        true_pool = list(true_anomalies)
        random.shuffle(true_pool)
        correct_desc = anomaly_lookup.get(true_pool[0], true_pool[0])

    entries: List[Tuple[str, bool]] = [(correct_desc, True)]
    if not is_normal:
        entries.append((_NO_ANOMALY_DESC, False))
    while len(entries) < 4 and distractor_pool:
        name = distractor_pool.pop()
        entries.append((anomaly_lookup.get(name, name), False))

    random.shuffle(entries)
    entries = entries[:4]

    labels = ["A", "B", "C", "D"]
    options: Dict[str, str] = {}
    answer = "A"
    for i, (desc, is_correct) in enumerate(entries):
        options[labels[i]] = desc
        if is_correct:
            answer = labels[i]
    return options, answer


_COMPARATIVE_OPTION_ORDER = ["mc_020", "mc_022", "mc_023", "mc_026"]


def _modal_phase(rows: List[Dict[str, Any]]) -> Optional[str]:
    phases = [
        str(r.get("task_phase"))
        for r in rows
        if r.get("task_phase") not in (None, "None", "")
    ]
    if not phases:
        return None
    return Counter(phases).most_common(1)[0][0]


def build_comparative_multi_select(
    fault_a: int,
    fault_b: int,
    machine_id_a: int,
    machine_id_b: int,
    task_id_a: str,
    task_id_b: str,
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
    mc_lookup: Dict[str, str],
) -> Optional[Tuple[Dict[str, str], str]]:
    """Build fixed-order options + TFFT answer for template 3.

    Options are always presented in the order
    [mc_020, mc_022, mc_023, mc_026] (labels A..D) and the answer encodes
    whether each corresponding statement holds, independently:
      - mc_020: different robots
      - mc_022: different anomalous states
      - mc_023: different tasks
      - mc_026: same task but different modal phase

    Returns None when no statement is true (no meaningful change).
    """
    if not mc_lookup:
        return None

    option_ids = [normalize_mc_option_id(x) for x in _COMPARATIVE_OPTION_ORDER]
    statements = [mc_lookup.get(oid) for oid in option_ids]
    if not all(statements):
        return None

    different_robots = machine_id_a != machine_id_b
    different_anomalous_state = (fault_a == 0) != (fault_b == 0) or (
        fault_a != 0 and fault_b != 0 and fault_a != fault_b
    )
    different_tasks = task_id_a != task_id_b

    mode_a = _modal_phase(rows_a)
    mode_b = _modal_phase(rows_b)
    same_task_different_phases = (
        (not different_tasks)
        and mode_a is not None
        and mode_b is not None
        and mode_a != mode_b
    )

    flags = [
        different_robots,
        different_anomalous_state,
        different_tasks,
        same_task_different_phases,
    ]
    if not any(flags):
        return None

    labels = ["A", "B", "C", "D"]
    options: Dict[str, str] = {lbl: stmt for lbl, stmt in zip(labels, statements)}
    answer = "".join("T" if f else "F" for f in flags)
    return options, answer


def build_severity_ranking(
    segments: List[Tuple[List[Dict[str, Any]], int]],
    root_causes: Dict[int, Dict[str, Any]],
    important_features: Optional[List[str]] = None,
    min_chunk: int = 5,
    max_chunk: int = 7,
) -> Optional[Tuple[Dict[str, str], str]]:
    """
    Build ranking options + answer string for template 5.
    Each option is a short encoded chunk from one episode filtered by important_features.
    Answer ranks options from most to least severe.
    """
    keep = set(important_features) if important_features else None
    labels = ["A", "B", "C", "D"]
    labeled: List[Tuple[str, str, int]] = []

    for i, (rows, fault_label) in enumerate(segments[:4]):
        rc = root_causes.get(fault_label, root_causes.get(0, {}))
        srank = get_severity_rank(fault_label, rc)

        chunk_len = random.randint(min_chunk, min(max_chunk, len(rows)))
        if len(rows) < chunk_len:
            return None
        start = random.randint(0, len(rows) - chunk_len)
        chunk = rows[start : start + chunk_len]
        stripped = [
            {k: v for k, v in r.items() if k != "timestamp_ms" and (keep is None or k in keep)}
            for r in chunk
        ]
        encoded = encode_chunk(stripped)
        labeled.append((labels[i], encoded, srank))

    if len(labeled) < 4:
        return None

    sorted_by_severity = sorted(labeled, key=lambda x: x[2], reverse=True)
    answer = "".join(x[0] for x in sorted_by_severity)
    options = {label: encoded for label, encoded, _ in labeled}
    return options, answer


# ---------------------------------------------------------------------------
# Template filling
# ---------------------------------------------------------------------------


def fill_template(
    template: Dict[str, Any],
    rows: List[Dict[str, Any]],
    root_causes: Dict[int, Dict[str, Any]],
    anomaly_lookup: Dict[str, str],
    mc_option_lookup: Dict[str, str],
    rows_b: Optional[List[Dict[str, Any]]] = None,
    machine_id: int = -1,
    machine_id_b: int = -1,
    task_id: str = "",
    task_id_b: str = "",
    severity_segments: Optional[List[Tuple[List[Dict[str, Any]], int]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fill a Level 1 question template.

    Template IDs:
      1 - phase segmentation (numerical)
      2 - anomaly detection (single-select)
      3 - comparative change detection (multi-select, TFFT)
      5 - severity ranking
      6 - robot identification (single-select)
    """
    tid = template["id"]
    tmpl_text: str = template["template"]
    answer_format: Dict[str, Any] = template["answer_format"]

    fault_label = pick_fault_label(rows)
    options: Dict[str, Any] = {}
    answer: Any = None
    acceptance_bounds = None

    if tid == 1:
        # Find contiguous phase segments in the subseries
        all_phases: List[Tuple[str, int, int]] = []  # (phase_name, start_idx, length)
        current_phase = None
        phase_start = 0
        for i, row in enumerate(rows):
            p = row.get("task_phase")
            if p != current_phase:
                if current_phase is not None and str(current_phase) not in ("None", "none", ""):
                    all_phases.append((str(current_phase), phase_start, i - phase_start))
                current_phase = p
                phase_start = i
        if current_phase is not None and str(current_phase) not in ("None", "none", ""):
            all_phases.append((str(current_phase), phase_start, len(rows) - phase_start))

        # Exclude first and last phases, filter to at least 3 timesteps
        if len(all_phases) < 3:
            return None
        inner_phases = [
            (name, start, length)
            for name, start, length in all_phases[1:-1]
            if length >= 3
        ]
        if not inner_phases:
            return None

        phase_name, phase_start_idx, phase_length = random.choice(inner_phases)
        window_length = phase_length + 5
        answer = phase_start_idx
        _TASK_DISPLAY = {"pick_and_place": "pick-and-place", "screwing": "screwing", "peg_in_hole": "peg-in-hole"}
        task_display = _TASK_DISPLAY.get(task_id, task_id.replace("_", " ") if task_id else "manipulation")
        question = fill(tmpl_text, task=task_display, phase=_phase_display_name(phase_name, task_id), window_length=window_length)
        acceptance_bounds = {"tolerance": 3}

    elif tid == 2:
        if not anomaly_lookup:
            return None
        options, answer = build_anomaly_single_select(fault_label, root_causes, anomaly_lookup)
        question = tmpl_text

    elif tid == 3:
        if rows_b is None:
            return None
        fault_b = pick_fault_label(rows_b)
        result = build_comparative_multi_select(
            fault_label, fault_b, machine_id, machine_id_b,
            task_id, task_id_b, rows, rows_b, mc_option_lookup,
        )
        if result is None:
            return None
        options, answer = result
        question = tmpl_text

    elif tid == 5:
        if not severity_segments or len(severity_segments) < 4:
            return None
        important_features = template.get("important_features")
        result = build_severity_ranking(
            severity_segments, root_causes, important_features=important_features
        )
        if result is None:
            return None
        options, answer = result
        question = tmpl_text

    elif tid == 6:
        # Robot identification: answer is determined by machine_id
        robot_names = {
            0: "Universal Robots UR3e",
            2: "Agile Robots Yu 5 Industrial",
            3: "KUKA KR 10 R1100-2",
        }
        if machine_id not in robot_names:
            return None
        correct_name = robot_names[machine_id]
        fixed = answer_format.get("fixed_options", list(robot_names.values()))
        labels = ["A", "B", "C"][:len(fixed)]
        entries = list(fixed)
        random.shuffle(entries)
        options = {}
        answer = "A"
        for i, name in enumerate(entries):
            options[labels[i]] = name
            if name == correct_name:
                answer = labels[i]
        question = tmpl_text

    elif tid == 7:
        # Prediction: given the subseries, predict a signal value n steps ahead
        important_features = template.get("important_features", [])
        # Pick a signal that exists and has numeric values
        candidate_signals = [
            s for s in important_features
            if any(isinstance(row.get(s), (int, float)) for row in rows[-5:])
        ]
        if not candidate_signals:
            return None
        signal = random.choice(candidate_signals)
        # steps_ahead: how many steps beyond the given context
        steps_ahead = random.randint(1, 10)
        # The rows passed in are the context; we need the actual future value
        # which is stored in kwargs via the generation loop
        # For now, store steps_ahead and signal; the generation loop provides the answer
        question = fill(tmpl_text, signal=_signal_display_name(signal), n=steps_ahead)
        answer = None  # set by generation loop
        acceptance_bounds = {"signal": signal, "steps_ahead": steps_ahead}

    else:
        logger.warning(f"Unknown template id: {tid}")
        return None

    return {
        "question": question,
        "answer_format": answer_format,
        "options": options,
        "answer": answer,
        "acceptance_bounds": acceptance_bounds,
    }


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def generate_level1_questions(
    datasets_dir: Path,
    output_dir: Path,
    templates: List[Dict[str, Any]],
    root_causes: Dict[int, Dict[str, Any]],
    anomaly_lookup: Dict[str, str],
    mc_option_lookup: Dict[str, str],
    dataset_index: Dict[str, Dict[str, Any]],
    n: int = 100,
    seed: Optional[int] = None,
    datasets: Optional[List[str]] = None,
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    allowed = datasets if datasets else VALID_DATASETS
    by_dataset = discover_episodes_by_dataset(datasets_dir, allowed)
    episodes_by_dataset = {ds: paths for ds, paths in by_dataset.items() if paths}
    available_datasets = list(episodes_by_dataset.keys())
    if not available_datasets:
        raise FileNotFoundError(
            f"No usable datasets found under {datasets_dir / 'normalized_episodes'} "
            f"for datasets: {allowed}"
        )

    all_episode_paths: List[Tuple[str, Path]] = [
        (ds, ep) for ds, eps in episodes_by_dataset.items() for ep in eps
    ]

    episode_cache: Dict[str, List[Dict[str, Any]]] = {}
    _normal_cache: Dict[str, bool] = {}

    def load_episode(path: Path) -> List[Dict[str, Any]]:
        key = str(path)
        if key not in episode_cache:
            raw = load_json(path)
            if isinstance(raw, dict):
                raw = raw.get("baseline", raw.get("flat", []))
            episode_cache[key] = raw
        return episode_cache[key]

    def is_normal_episode(rows: List[Dict[str, Any]], path: Path) -> bool:
        """Return True if the episode has no anomalies (fault_label is 0 or absent throughout)."""
        key = str(path)
        if key in _normal_cache:
            return _normal_cache[key]
        result = all(
            int(row.get("fault_label", 0) or 0) == 0
            for row in rows
            if row.get("fault_label") is not None
        )
        _normal_cache[key] = result
        return result

    # Exclude anomaly-related templates for now (IDs 2 and 5)
    _DISABLED_TEMPLATE_IDS = {2, 5}
    usable_templates = [t for t in templates if t["id"] not in _DISABLED_TEMPLATE_IDS]
    if not usable_templates:
        raise ValueError("No usable templates for Level 1 generation.")

    generated = 0
    attempts = 0
    max_total_attempts = n * 20

    while generated < n and attempts < max_total_attempts:
        attempts += 1
        template = random.choice(usable_templates)
        tid = template["id"]
        important_features = template.get("important_features")

        # ------------------------------------------------------------------
        # Template 1: single episode, full time series as context
        # ------------------------------------------------------------------
        if tid == 1:
            _MAX_EPISODE_STEPS = 64
            ds = random.choice(available_datasets)
            ep_path = random.choice(episodes_by_dataset[ds])
            rows = load_episode(ep_path)
            if not isinstance(rows, list) or len(rows) < CONTEXT_MIN:
                continue

            full_rows = normalize_timestamps(rows, _first_timestamp_ms(rows))

            # Anti-aliased downsample to at most _MAX_EPISODE_STEPS
            downsample_factor = 1
            if len(full_rows) > _MAX_EPISODE_STEPS:
                downsample_factor = math.ceil(len(full_rows) / _MAX_EPISODE_STEPS)
                _NON_CONTINUOUS = {"timestamp_ms", "fault_label", "task_phase"}
                df = pd.DataFrame(full_rows)
                continuous = set(df.columns) - _NON_CONTINUOUS
                df = decimate_dataframe(df, q=downsample_factor, continuous_cols=continuous)
                full_rows = df.where(df.notna(), None).to_dict(orient="records")

            # Try to determine task from metadata file
            meta_path = ep_path.with_name(ep_path.stem + "_metadata.json")
            ep_task = ""
            if meta_path.exists():
                try:
                    meta = load_json(meta_path)
                    ep_task = meta.get("task", "")
                except Exception:
                    pass

            filled = fill_template(
                template, full_rows, root_causes, anomaly_lookup, mc_option_lookup,
                machine_id=DATASET_MACHINE_ID.get(ds, -1),
                task_id=ep_task,
            )
            if filled is None:
                continue

            if important_features:
                keep = set(important_features) | {"timestamp_ms", "fault_label", "task_phase"}
                context_rows = [{k: v for k, v in row.items() if k in keep} for row in full_rows]
            else:
                context_rows = full_rows
            context = build_context(context_rows)

            item = {
                "id": str(uuid.uuid4()),
                "level": 1,
                "template_id": tid,
                "template_type": template["type"],
                "hides": template.get("hides", []),
                "question": filled["question"],
                "options": filled["options"],
                "answer": filled["answer"],
                "acceptance_bounds": filled.get("acceptance_bounds"),
                "provenance": {
                    "dataset": ds,
                    "episode": ep_path.stem,
                    "episode_length": len(full_rows),
                    "downsample_factor": downsample_factor,
                },
                "context": context,
            }

        # ------------------------------------------------------------------
        # Templates 2 and 6: single episode, sampled subseries
        # ------------------------------------------------------------------
        elif tid in (2, 6):
            ds = random.choice(available_datasets)
            ep_path = random.choice(episodes_by_dataset[ds])
            rows = load_episode(ep_path)
            if not isinstance(rows, list) or len(rows) < CONTEXT_MIN:
                continue

            sampled = sample_subseries(rows)
            if sampled is None:
                continue
            subseries, start_idx = sampled
            subseries = normalize_timestamps(subseries, _first_timestamp_ms(subseries))

            filled = fill_template(
                template, subseries, root_causes, anomaly_lookup, mc_option_lookup,
                machine_id=DATASET_MACHINE_ID.get(ds, -1),
            )
            if filled is None:
                continue

            if important_features:
                keep = set(important_features) | {"timestamp_ms", "fault_label", "task_phase"}
                context_rows = [{k: v for k, v in row.items() if k in keep} for row in subseries]
            else:
                context_rows = subseries
            context = build_context(context_rows)

            item = {
                "id": str(uuid.uuid4()),
                "level": 1,
                "template_id": tid,
                "template_type": template["type"],
                "hides": template.get("hides", []),
                "question": filled["question"],
                "options": filled["options"],
                "answer": filled["answer"],
                "acceptance_bounds": filled.get("acceptance_bounds"),
                "provenance": {
                    "dataset": ds,
                    "episode": ep_path.stem,
                    "subseries_start_index": start_idx,
                    "subseries_length": len(subseries),
                },
                "context": context,
            }

        # ------------------------------------------------------------------
        # Template 7: prediction — sample subseries + future steps for answer
        # ------------------------------------------------------------------
        elif tid == 7:
            ds = random.choice(available_datasets)
            ep_path = random.choice(episodes_by_dataset[ds])
            rows = load_episode(ep_path)
            if not isinstance(rows, list) or len(rows) < CONTEXT_MIN + 10:
                continue

            # Sample a subseries leaving room for future steps
            max_start = len(rows) - CONTEXT_MIN - 10
            if max_start < 0:
                continue
            context_len = random.randint(CONTEXT_MIN, min(CONTEXT_MAX, len(rows) - 10))
            start_idx = random.randint(0, len(rows) - context_len - 10)
            subseries = rows[start_idx:start_idx + context_len]
            subseries = normalize_timestamps(subseries, _first_timestamp_ms(subseries))

            filled = fill_template(
                template, subseries, root_causes, anomaly_lookup, mc_option_lookup,
                machine_id=DATASET_MACHINE_ID.get(ds, -1),
            )
            if filled is None:
                continue

            # Get the actual future value for the answer
            bounds = filled.get("acceptance_bounds", {})
            signal = bounds.get("signal")
            steps_ahead = bounds.get("steps_ahead", 1)
            future_idx = start_idx + context_len + steps_ahead - 1
            if future_idx >= len(rows) or signal is None:
                continue
            future_val = rows[future_idx].get(signal)
            if future_val is None or not isinstance(future_val, (int, float)):
                continue
            filled["answer"] = round(float(future_val), 4)
            filled["acceptance_bounds"]["actual_value"] = filled["answer"]

            if important_features:
                keep = set(important_features) | {"timestamp_ms", "fault_label", "task_phase"}
                context_rows = [{k: v for k, v in row.items() if k in keep} for row in subseries]
            else:
                context_rows = subseries
            context = build_context(context_rows)

            item = {
                "id": str(uuid.uuid4()),
                "level": 1,
                "template_id": tid,
                "template_type": template["type"],
                "hides": template.get("hides", []),
                "question": filled["question"],
                "options": filled["options"],
                "answer": filled["answer"],
                "acceptance_bounds": filled.get("acceptance_bounds"),
                "provenance": {
                    "dataset": ds,
                    "episode": ep_path.stem,
                    "subseries_start_index": start_idx,
                    "subseries_length": context_len,
                    "prediction_index": future_idx,
                },
                "context": context,
            }

        # ------------------------------------------------------------------
        # Template 3: two episodes (prefer different datasets)
        # ------------------------------------------------------------------
        elif tid == 3:
            if len(available_datasets) >= 2:
                ds_a, ds_b = random.sample(available_datasets, 2)
            else:
                ds_a = ds_b = available_datasets[0]

            ep_a = random.choice(episodes_by_dataset[ds_a])
            ep_b = random.choice(episodes_by_dataset[ds_b])
            rows_a = load_episode(ep_a)
            rows_b_raw = load_episode(ep_b)

            if not (isinstance(rows_a, list) and isinstance(rows_b_raw, list)):
                continue
            if len(rows_a) < CONTEXT_MIN or len(rows_b_raw) < CONTEXT_MIN:
                continue

            sampled_a = sample_subseries(rows_a)
            sampled_b = sample_subseries(rows_b_raw)
            if sampled_a is None or sampled_b is None:
                continue

            sub_a, start_a = sampled_a
            sub_b, start_b = sampled_b
            sub_a = normalize_timestamps(sub_a, _first_timestamp_ms(sub_a))
            sub_b = normalize_timestamps(sub_b, _first_timestamp_ms(sub_b))

            filled = fill_template(
                template, sub_a, root_causes, anomaly_lookup, mc_option_lookup,
                rows_b=sub_b,
                machine_id=DATASET_MACHINE_ID.get(ds_a, -1),
                machine_id_b=DATASET_MACHINE_ID.get(ds_b, -1),
                task_id=dataset_index.get(ds_a, {}).get("task_id", ""),
                task_id_b=dataset_index.get(ds_b, {}).get("task_id", ""),
            )
            if filled is None:
                continue

            if important_features:
                keep = set(important_features) | {"timestamp_ms", "fault_label", "task_phase"}
                sub_a_ctx = [{k: v for k, v in row.items() if k in keep} for row in sub_a]
                sub_b_ctx = [{k: v for k, v in row.items() if k in keep} for row in sub_b]
            else:
                sub_a_ctx, sub_b_ctx = sub_a, sub_b

            context = {
                "series_a": build_context(sub_a_ctx),
                "series_b": build_context(sub_b_ctx),
            }

            item = {
                "id": str(uuid.uuid4()),
                "level": 1,
                "template_id": tid,
                "template_type": template["type"],
                "hides": template.get("hides", []),
                "question": filled["question"],
                "options": filled["options"],
                "answer": filled["answer"],
                "acceptance_bounds": filled.get("acceptance_bounds"),
                "provenance": {
                    "dataset_a": ds_a,
                    "machine_id_a": DATASET_MACHINE_ID.get(ds_a, -1),
                    "episode_a": ep_a.stem,
                    "subseries_start_a": start_a,
                    "dataset_b": ds_b,
                    "machine_id_b": DATASET_MACHINE_ID.get(ds_b, -1),
                    "episode_b": ep_b.stem,
                    "subseries_start_b": start_b,
                },
                "context": context,
            }

        # ------------------------------------------------------------------
        # Template 5: four episodes for severity ranking
        # ------------------------------------------------------------------
        elif tid == 5:
            if len(all_episode_paths) < 4:
                continue
            sampled_eps = random.sample(all_episode_paths, 4)

            segments: List[Tuple[List[Dict[str, Any]], int]] = []
            valid = True
            for ds, ep_path in sampled_eps:
                ep_rows = load_episode(ep_path)
                if not isinstance(ep_rows, list) or len(ep_rows) < 5:
                    valid = False
                    break
                fl = pick_fault_label(ep_rows)
                segments.append((ep_rows, fl))
            if not valid or len(segments) < 4:
                continue

            filled = fill_template(
                template, segments[0][0], root_causes, anomaly_lookup, mc_option_lookup,
                severity_segments=segments,
            )
            if filled is None:
                continue

            item = {
                "id": str(uuid.uuid4()),
                "level": 1,
                "template_id": tid,
                "template_type": template["type"],
                "hides": template.get("hides", []),
                "question": filled["question"],
                "options": filled["options"],
                "answer": filled["answer"],
                "acceptance_bounds": filled.get("acceptance_bounds"),
                "provenance": {
                    "episodes": [
                        {"dataset": ds, "episode": ep.stem}
                        for ds, ep in sampled_eps
                    ],
                },
                "context": {},
            }

        else:
            continue

        out_path = output_dir / f"level1_{generated:04d}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)

        logger.info(f"✓ [{generated + 1}/{n}] {out_path.name} (template {tid})")
        generated += 1

    if generated < n:
        logger.warning(f"Only generated {generated}/{n} questions after {attempts} attempts.")
    else:
        logger.info(f"Done: {generated} questions written to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Level 1 (State Understanding) Q&A pairs."
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
        default=repo_root / "output" / "questions" / "level1",
        help="Output directory (default: <repo>/output/questions/level1)",
    )
    parser.add_argument("-n", type=int, default=100, help="Number of questions to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Datasets to sample from (default: all). Choices: {VALID_DATASETS}",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    templates = load_templates(Path(__file__).with_name("question_template.json"))
    root_causes = load_root_causes(args.datasets_dir / "labelling" / "rca" / "root_causes.json")

    global _ANOMALY_RANKING
    _ANOMALY_RANKING = load_anomaly_ranking(
        args.datasets_dir / "labelling" / "rca" / "anomaly_ranking.json"
    )

    anomaly_lookup = load_anomaly_lookup(
        args.datasets_dir / "labelling" / "rca" / "anomalies.json"
    )
    mc_option_lookup = load_mc_option_lookup(
        args.datasets_dir / "mc_options" / "mc_options.json",
        level=1,
    )
    dataset_index = load_dataset_index(args.datasets_dir / "labelling" / "dataset.json")

    generate_level1_questions(
        datasets_dir=args.datasets_dir,
        output_dir=args.output,
        templates=templates,
        root_causes=root_causes,
        anomaly_lookup=anomaly_lookup,
        mc_option_lookup=mc_option_lookup,
        dataset_index=dataset_index,
        n=args.n,
        seed=args.seed,
        datasets=args.datasets,
    )


if __name__ == "__main__":
    main()
