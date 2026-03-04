"""
Level 3 question generator: Root Cause Analysis.

Reads normalized episode JSON files from aursad or vorausad datasets,
samples random sub-series, and fills Level 3 question templates.
Answers are left null (to be filled by annotation).

Output: datasets/questions/level3/level3_{NNNN}.json

Usage:
    python -m src.questions.level3.level3 -n 100 --seed 42

Template IDs (from question_template.json):
  1 - signal_segment_ranking        chunks embedded in text + options dict
  2 - intervention_outcome          MC, options null
  3 - saturation_prediction         no event; velocity/torque/override from data
  4 - trajectory_outcome_multiselect choices embedded in text, options null
  5 - signal_value_prediction       numerical
  6 - signal_value_prediction       tensor (6-element)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.questions.common.io import load_events, load_json, load_root_causes, load_templates
from src.questions.common.template import (
    build_context,
    discover_episodes_by_dataset,
    encode_chunk,
    fill,
    fill_event_description,
    get_last_timestamp,
    pick_joint_velocity_and_torque,
    pick_scalar_signal,
    sample_chunks,
)
from src.questions.common.time_series import (
    is_inactive_subseries,
    parse_event_id,
    pick_fault_label,
    sample_subseries_before_event,
)

logger = logging.getLogger(__name__)

VALID_DATASETS = ["test_aursad", "test_vorausad"]
OVERRIDE_PCTS = [110, 120, 130, 150, 175, 200]
ROBOT_ACTIONS = ["pick", "place", "screwing", "move", "approach", "retract"]
PREDICTION_HORIZONS_MS = [50, 100, 250, 500, 1000]

TRAJECTORY_EXTRA_STATEMENTS = [
    "The robot arm is operating within its nominal torque limits.",
    "At least one joint has exceeded its velocity setpoint.",
    "The TCP force is below detection threshold.",
    "The motor current has stabilized.",
    "A protective stop is imminent.",
    "The control loop has lost tracking.",
    "Vibration levels are within normal range.",
    "The gripper command is mismatched with the current phase.",
]


# ---------------------------------------------------------------------------
# Template filling  (level 3 specific)
# ---------------------------------------------------------------------------


def fill_template(
    template: Dict[str, Any],
    subseries: List[Dict[str, Any]],
    post_event_rows: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Fill a Level 3 question template.

    post_event_rows: rows from the episode after the subseries end
                     (used for ranking chunks).

    Returns dict with question/options/answer_format/event_id, or None on failure.

    Template IDs:
      1 - signal_segment_ranking        : chunks in both text and options
      2 - intervention_outcome          : MC, options null
      3 - saturation_prediction         : no event; data-derived velocity/torque
      4 - trajectory_outcome_multiselect: choices embedded in text, options null
      5 - signal_value_prediction (numerical)
      6 - signal_value_prediction (tensor)
    """
    tid = template["id"]
    tmpl_text: str = template["template"]
    answer_format: Dict[str, Any] = template["answer_format"]
    t = get_last_timestamp(subseries)

    onset_id = parse_event_id(post_event_rows[0].get("event", 0)) if post_event_rows else 0
    event_obj = next((e for e in events if e["id"] == onset_id), random.choice(events))
    event_id = event_obj["id"]
    event_desc = fill_event_description(event_obj, subseries, t, post_event_rows)

    options = None

    if tid == 1:
        chunks = sample_chunks(post_event_rows, n_chunks=4, min_chunk=5, max_chunk=7)
        if len(chunks) < 4:
            return None
        random.shuffle(chunks)
        labels = ["A", "B", "C", "D"]
        encoded = [encode_chunk(chunks[i]) for i in range(4)]
        options = {label: encoded[i] for i, label in enumerate(labels)}
        question = fill(
            tmpl_text,
            t=t,
            event=event_desc,
            chunk_a=encoded[0],
            chunk_b=encoded[1],
            chunk_c=encoded[2],
            chunk_d=encoded[3],
        )

    elif tid == 2:
        question = fill(tmpl_text, event=event_desc, t=t)

    elif tid == 3:
        # saturation_prediction — no event, uses kinematic data from subseries
        override_pct = random.choice(OVERRIDE_PCTS)
        action = random.choice(ROBOT_ACTIONS)
        velocity, torque = pick_joint_velocity_and_torque(subseries)
        if velocity is None:
            velocity = round(random.uniform(0.1, 2.0), 3)
        if torque is None:
            torque = round(random.uniform(0.5, 50.0), 3)
        question = fill(
            tmpl_text,
            override_pct=override_pct,
            action=action,
            velocity=velocity,
            torque=torque,
        )

    elif tid == 4:
        fixed = answer_format.get("fixed_statements", [])
        extra = random.sample(
            TRAJECTORY_EXTRA_STATEMENTS,
            min(2, len(TRAJECTORY_EXTRA_STATEMENTS)),
        )
        all_statements = fixed + extra
        random.shuffle(all_statements)
        choices_str = "\n".join(f"- {s}" for s in all_statements)
        question = fill(tmpl_text, event=event_desc, t=t, choices=choices_str)

    elif tid in (5, 6):
        signal = pick_scalar_signal(subseries)
        if signal is None:
            return None
        n_ms = random.choice(PREDICTION_HORIZONS_MS)
        question = fill(tmpl_text, event=event_desc, t=t, signal=signal, n=n_ms)

    else:
        logger.warning(f"Unknown template id: {tid}")
        return None

    return {
        "question": question,
        "answer_format": answer_format,
        "options": options,
        "event_id": event_id,
    }


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def generate_level3_questions(
    datasets_dir: Path,
    output_dir: Path,
    templates: List[Dict[str, Any]],
    root_causes: Dict[int, Dict[str, Any]],
    events: List[Dict[str, Any]],
    n: int = 100,
    min_len: int = 32,
    max_len: int = 64,
    seed: Optional[int] = None,
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    by_dataset = discover_episodes_by_dataset(datasets_dir, VALID_DATASETS)
    if not by_dataset:
        raise FileNotFoundError(
            f"No normalized episode JSON files found under "
            f"{datasets_dir / 'normalized_episodes'} for datasets: {VALID_DATASETS}"
        )

    available_datasets = list(by_dataset.keys())
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

        ds = random.choice(available_datasets)
        ep_path = random.choice(by_dataset[ds])
        rows = load_episode(ep_path)
        if not isinstance(rows, list) or len(rows) < min_len:
            continue

        subseries, post_event_rows = sample_subseries_before_event(rows, min_len, max_len)
        if not subseries:
            continue

        if is_inactive_subseries(subseries):
            logger.debug(f"Inactive subseries from {ep_path.name}, skipping")
            continue

        fault_label = pick_fault_label(subseries)
        template = random.choice(templates)

        filled = fill_template(template, subseries, post_event_rows, events)
        if filled is None:
            continue

        item = {
            "id": str(uuid.uuid4()),
            "level": 3,
            "template_id": template["id"],
            "template_type": template["type"],
            "question": filled["question"],
            "options": filled["options"],
            "answer_format": filled["answer_format"],
            "answer": None,
            "provenance": {
                "dataset": ds,
                "episode": ep_path.stem,
                "fault_label": fault_label,
                "event_id": filled["event_id"],
            },
            "context": build_context(subseries),
        }

        out_path = output_dir / f"level3_{generated:04d}.json"
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
        description="Generate Level 3 (Root Cause Analysis) Q&A pairs."
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
        default=repo_root / "data" / "questions" / "level3",
        help="Output directory (default: <repo>/data/questions/level3)",
    )
    parser.add_argument("-n", type=int, default=100, help="Number of questions to generate")
    parser.add_argument("--min-len", type=int, default=32, help="Min subseries length")
    parser.add_argument("--max-len", type=int, default=64, help="Max subseries length")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    templates = load_templates(Path(__file__).with_name("question_template.json"))
    root_causes = load_root_causes(args.datasets_dir / "rca" / "root_causes.json")
    events = load_events(args.datasets_dir / "events" / "events.json")

    generate_level3_questions(
        datasets_dir=args.datasets_dir,
        output_dir=args.output,
        templates=templates,
        root_causes=root_causes,
        events=events,
        n=args.n,
        min_len=args.min_len,
        max_len=args.max_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
