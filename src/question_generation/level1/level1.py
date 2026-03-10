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
import random
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.question_generation.utils.io import load_json, load_root_causes, load_templates
from src.question_generation.utils.template import (
    build_context,
    discover_episodes_by_dataset,
    encode_chunk,
    fill,
)
from src.question_generation.utils.time_series import (
    encode_time_series,
    format_note_value,
    pick_fault_label,
    remove_constant_features,
    remove_feature,
    sort_feature_keys,
    strip_null_features,
)

logger = logging.getLogger(__name__)

VALID_DATASETS = ["inter_aursad", "inter_vorausad"]
SEVERITY_ORDER: Dict[str, int] = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

# Robot name associated with each dataset, used in template 3 provenance
DATASET_ROBOT: Dict[str, str] = {
    "inter_aursad": "UR3e",
    "inter_vorausad": "UR5",
}

# Columns kept in the context for template 6 (setpoint-only)
_SETPOINT_PREFIXES = ("setpoint_pos_", "setpoint_speed_")

# Synthetic "no anomaly" option for template 2
_NO_ANOMALY_ID = "no_anomaly"
_NO_ANOMALY_DESC = "No anomaly is present; the machine is operating nominally."

# Subseries length bounds (shared across templates 2 and 6)
CONTEXT_MIN = 16
CONTEXT_MAX = 64


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


def get_severity_rank(root_cause: Dict[str, Any]) -> int:
    """Return the maximum severity integer for a root-cause dict."""
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


def load_mc_option_lookup(
    path: Path,
    level: int,
    template_id: Optional[int] = None,
) -> Dict[str, str]:
    """Load MC options filtered by level and optionally by template_id."""
    try:
        raw = load_json(path)
    except Exception as exc:
        logger.warning(f"Could not load MC options from {path}: {exc}")
        return {}
    if not isinstance(raw, list):
        return {}

    lookup: Dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        allowed = item.get("usable_levels")
        if isinstance(allowed, list):
            try:
                parsed = [int(v) for v in allowed]
            except (TypeError, ValueError):
                parsed = []
            if parsed and level not in parsed:
                continue
        if template_id is not None:
            tid = item.get("template_id")
            if tid is not None and tid != template_id:
                continue
        oid = item.get("id")
        stmt = item.get("statement")
        if isinstance(oid, str) and isinstance(stmt, str) and stmt.strip():
            lookup[normalize_mc_option_id(oid)] = stmt.strip()
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


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def build_setpoint_context(subseries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a context that includes only setpoint_pos and setpoint_speed columns."""
    filtered = [
        {k: v for k, v in row.items() if k == "timestamp_ms" or k.startswith(_SETPOINT_PREFIXES)}
        for row in subseries
    ]
    ts = strip_null_features(filtered)
    ts = sort_feature_keys(ts)
    ts, constant_features = remove_constant_features(ts)
    encoded, acronym_mapping = encode_time_series(ts)

    ctx: Dict[str, Any] = {
        "time_series_format": {
            "description": (
                "Each row is one timestep encoded as 't=<timestamp>: acronym=value, ...'. "
                "Only setpoint signals (position and velocity) are provided. "
                "Feature names use acronyms defined in provenance.feature_mapping."
            ),
            "acronym_mapping": acronym_mapping,
        },
        "time_series": encoded,
    }
    if constant_features:
        ctx["notes"] = {
            "disclaimer": "these features stayed constant at the following values",
            "constant_features": {
                k: format_note_value(constant_features[k])
                for k in sorted(constant_features.keys())
            },
        }
    return ctx


def build_paired_context(
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a context with two independently encoded time series for template 3."""
    return {
        "series_a": build_context(rows_a),
        "series_b": build_context(rows_b),
    }


# ---------------------------------------------------------------------------
# Answer builders (one per template)
# ---------------------------------------------------------------------------


def build_anomaly_multiselect(
    fault_label: int,
    root_causes: Dict[int, Dict[str, Any]],
    anomaly_lookup: Dict[str, str],
) -> Tuple[Dict[str, str], str]:
    """
    Build 4 multi-select options + T/F answer string for template 2.

    Includes true anomaly(ies) from the root cause, random distractors from
    the anomaly vocabulary, and always one 'no anomaly' option.
    """
    root_cause = root_causes.get(fault_label, root_causes.get(0, {}))
    true_anomalies: List[str] = root_cause.get("possible_anomalies", [])
    is_normal = fault_label == 0 or not true_anomalies

    # Draw up to 2 true anomalies
    true_pool = list(true_anomalies)
    random.shuffle(true_pool)
    selected_true = true_pool[:min(2, len(true_pool))]

    # Distractor pool: anomalies not associated with this root cause
    distractor_pool = [a for a in anomaly_lookup if a not in true_anomalies]
    random.shuffle(distractor_pool)

    entries: List[Tuple[str, bool]] = []
    for name in selected_true:
        entries.append((anomaly_lookup.get(name, name), True))
    # Fill to 3 with distractors, then add the "no anomaly" option
    while len(entries) < 3 and distractor_pool:
        name = distractor_pool.pop()
        entries.append((anomaly_lookup.get(name, name), False))
    entries.append((_NO_ANOMALY_DESC, is_normal))

    random.shuffle(entries)
    entries = entries[:4]

    labels = ["A", "B", "C", "D"]
    options: Dict[str, str] = {}
    answer_chars: List[str] = []
    for i, (desc, is_true) in enumerate(entries):
        options[labels[i]] = desc
        answer_chars.append("T" if is_true else "F")
    return options, "".join(answer_chars)


def build_comparative_single_select(
    fault_a: int,
    fault_b: int,
    dataset_a: str,
    dataset_b: str,
    mc_lookup: Dict[str, str],
) -> Optional[Tuple[Dict[str, str], str]]:
    """
    Build 4 single-select options + answer letter for template 3.
    Determines what actually changed between the two episodes.
    """
    if not mc_lookup:
        return None

    different_robots = dataset_a != dataset_b
    different_anomalous_state = (fault_a == 0) != (fault_b == 0) or (
        fault_a != 0 and fault_b != 0 and fault_a != fault_b
    )

    if different_robots:
        correct_id = normalize_mc_option_id("mc_020")
    elif different_anomalous_state:
        correct_id = normalize_mc_option_id("mc_022")
    else:
        correct_id = normalize_mc_option_id("mc_022")

    correct_stmt = mc_lookup.get(correct_id)
    if not correct_stmt:
        return None

    distractor_ids = [k for k in mc_lookup if k != correct_id]
    random.shuffle(distractor_ids)

    all_entries: List[Tuple[str, bool]] = [(correct_stmt, True)]
    for did in distractor_ids[:3]:
        all_entries.append((mc_lookup[did], False))
    random.shuffle(all_entries)

    labels = ["A", "B", "C", "D"]
    options: Dict[str, str] = {}
    answer = "A"
    for i, (stmt, is_correct) in enumerate(all_entries[:4]):
        options[labels[i]] = stmt
        if is_correct:
            answer = labels[i]
    return options, answer


def build_severity_ranking(
    segments: List[Tuple[List[Dict[str, Any]], int]],  # (rows, fault_label)
    root_causes: Dict[int, Dict[str, Any]],
    min_chunk: int = 5,
    max_chunk: int = 7,
) -> Optional[Tuple[Dict[str, str], str]]:
    """
    Build ranking options + answer string for template 5.
    Each option is a short encoded chunk from one episode.
    Answer ranks options from most to least severe.
    """
    labels = ["A", "B", "C", "D"]
    labeled: List[Tuple[str, str, int]] = []  # (label, encoded_chunk, severity)

    for i, (rows, fault_label) in enumerate(segments[:4]):
        rc = root_causes.get(fault_label, root_causes.get(0, {}))
        srank = get_severity_rank(rc)

        chunk_len = random.randint(min_chunk, min(max_chunk, len(rows)))
        if len(rows) < chunk_len:
            return None
        start = random.randint(0, len(rows) - chunk_len)
        chunk = rows[start : start + chunk_len]
        stripped = [{k: v for k, v in r.items() if k != "timestamp_ms"} for r in chunk]
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
    dataset: str = "",
    dataset_b: str = "",
    severity_segments: Optional[List[Tuple[List[Dict[str, Any]], int]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fill a Level 1 question template.

    Template IDs:
      1 - phase segmentation (numerical)     — skipped: requires phase_label column
      2 - anomaly detection (multi-select T/F)
      3 - comparative change detection (single-select)
      5 - severity ranking
      6 - effort prediction from setpoints (tensor)
    """
    tid = template["id"]
    tmpl_text: str = template["template"]
    answer_format: Dict[str, Any] = template["answer_format"]

    fault_label = pick_fault_label(rows)
    options: Dict[str, Any] = {}
    answer: Any = None

    if tid == 1:
        logger.debug("Template 1 skipped: phase_label column not present in current datasets.")
        return None

    elif tid == 2:
        if not anomaly_lookup:
            return None
        options, answer = build_anomaly_multiselect(fault_label, root_causes, anomaly_lookup)
        question = tmpl_text

    elif tid == 3:
        if rows_b is None:
            return None
        fault_b = pick_fault_label(rows_b)
        result = build_comparative_single_select(
            fault_label, fault_b, dataset, dataset_b, mc_option_lookup
        )
        if result is None:
            return None
        options, answer = result
        question = tmpl_text

    elif tid == 5:
        if not severity_segments or len(severity_segments) < 4:
            return None
        result = build_severity_ranking(severity_segments, root_causes)
        if result is None:
            return None
        options, answer = result
        question = tmpl_text

    elif tid == 6:
        joint = random.randint(0, 5)
        joint_key = f"effort_current_{joint}"
        n = len(rows)
        if n == 0:
            return None
        values: List[float] = []
        for row in rows:
            v = row.get(joint_key)
            if not isinstance(v, (int, float, np.floating)):
                return None
            values.append(round(float(v), 6))
        answer = "_".join(str(v) for v in values)
        question = fill(tmpl_text, joint=f"joint_{joint}", n=n)

    else:
        logger.warning(f"Unknown template id: {tid}")
        return None

    return {
        "question": question,
        "answer_format": answer_format,
        "options": options,
        "answer": answer,
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
    n: int = 100,
    seed: Optional[int] = None,
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    by_dataset = discover_episodes_by_dataset(datasets_dir, VALID_DATASETS)
    episodes_by_dataset = {ds: paths for ds, paths in by_dataset.items() if paths}
    available_datasets = list(episodes_by_dataset.keys())
    if not available_datasets:
        raise FileNotFoundError(
            f"No usable datasets found under {datasets_dir / 'normalized_episodes'} "
            f"for datasets: {VALID_DATASETS}"
        )

    all_episode_paths: List[Tuple[str, Path]] = [
        (ds, ep) for ds, eps in episodes_by_dataset.items() for ep in eps
    ]

    episode_cache: Dict[str, List[Dict[str, Any]]] = {}

    def load_episode(path: Path) -> List[Dict[str, Any]]:
        key = str(path)
        if key not in episode_cache:
            episode_cache[key] = load_json(path)
        return episode_cache[key]

    # Skip template 1 since phase_label is not available in current datasets
    usable_templates = [t for t in templates if t["id"] != 1]
    if not usable_templates:
        raise ValueError("No usable templates for Level 1 generation.")

    generated = 0
    attempts = 0
    max_total_attempts = n * 20

    while generated < n and attempts < max_total_attempts:
        attempts += 1
        template = random.choice(usable_templates)
        tid = template["id"]

        # ------------------------------------------------------------------
        # Template 2 and 6: single episode
        # ------------------------------------------------------------------
        if tid in (2, 6):
            ds = random.choice(available_datasets)
            ep_path = random.choice(episodes_by_dataset[ds])
            rows = load_episode(ep_path)
            if not isinstance(rows, list) or len(rows) < CONTEXT_MIN:
                continue

            sampled = sample_subseries(rows)
            if sampled is None:
                continue
            subseries, start_idx = sampled
            base_ts = _first_timestamp_ms(subseries)
            subseries = normalize_timestamps(subseries, base_ts)

            filled = fill_template(
                template, subseries, root_causes, anomaly_lookup, mc_option_lookup,
                dataset=ds,
            )
            if filled is None:
                continue

            context = build_setpoint_context(subseries) if tid == 6 else build_context(subseries)

            item = {
                "id": str(uuid.uuid4()),
                "level": 1,
                "template_id": tid,
                "template_type": template["type"],
                "question": filled["question"],
                "options": filled["options"],
                "answer": filled["answer"],
                "provenance": {
                    "dataset": ds,
                    "episode": ep_path.stem,
                    "subseries_start_index": start_idx,
                    "subseries_length": len(subseries),
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
                rows_b=sub_b, dataset=ds_a, dataset_b=ds_b,
            )
            if filled is None:
                continue

            context = build_paired_context(sub_a, sub_b)

            item = {
                "id": str(uuid.uuid4()),
                "level": 1,
                "template_id": tid,
                "template_type": template["type"],
                "question": filled["question"],
                "options": filled["options"],
                "answer": filled["answer"],
                "provenance": {
                    "dataset_a": ds_a,
                    "episode_a": ep_a.stem,
                    "subseries_start_a": start_a,
                    "dataset_b": ds_b,
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
                "question": filled["question"],
                "options": filled["options"],
                "answer": filled["answer"],
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

        logger.info(
            f"✓ [{generated + 1}/{n}] {out_path.name} (template {tid})"
        )
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
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    templates = load_templates(Path(__file__).with_name("question_template.json"))
    root_causes = load_root_causes(args.datasets_dir / "labelling" / "rca" / "root_causes.json")
    anomaly_lookup = load_anomaly_lookup(
        args.datasets_dir / "labelling" / "rca" / "anomalies.json"
    )
    mc_option_lookup = load_mc_option_lookup(
        args.datasets_dir / "mc_options" / "mc_options.json",
        level=1,
        template_id=3,
    )

    generate_level1_questions(
        datasets_dir=args.datasets_dir,
        output_dir=args.output,
        templates=templates,
        root_causes=root_causes,
        anomaly_lookup=anomaly_lookup,
        mc_option_lookup=mc_option_lookup,
        n=args.n,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
