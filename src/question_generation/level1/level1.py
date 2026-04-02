"""
Level 1 question generator: State identification.

Reads normalized episode JSON files from datasets,
samples random time windows across the entire sequence, and fills Level 1 question templates.
Answers are generated deterministically using raw values from episode readings.

Output: datasets/questions/level1/level1_{NNNN}.json

Usage:
    python -m src.question_generation.level1.level1 -n 100 --seed 27
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from src.question_generation.utils.io import load_json, load_templates
from src.question_generation.utils.template import build_context
from src.question_generation.level1.mc_truth import (
    answer_q1_state_joint_moved,
    answer_q2_state_friction_increase,
    answer_q3_state_acceleration,
    answer_q4_state_external_force_detected,
    answer_q5_state_signal_statistic,
    answer_q6_state_joint_speed_ranking,
    answer_q7_state_joint_within_rated_speed,
    answer_q8_state_current_within_rated,
    answer_q9_state_signal_description,
    answer_q10_state_safety_mode,
    get_num_joints,
    get_machine_by_id,
    load_machine_metadata
)

logger = logging.getLogger(__name__)

VALID_DATASETS = ["inter_aursad", "inter_voraus", "aursad", "voraus"]


def pick_time_window(rows: List[Dict[str, Any]], min_dt_ms: int, max_dt_ms: int) -> Tuple[int, int]:
    n = len(rows)
    if n < 2:
        raise ValueError("Not enough samples to pick a time window")
        
    valid_idx = [i for i, r in enumerate(rows) if r.get("timestamp_ms") is not None]
    if len(valid_idx) < 2:
        return 0, n - 1

    attempts = 0
    while attempts < 1000:
        i_pos = random.randint(0, len(valid_idx) - 2)
        i = valid_idx[i_pos]
        
        # Pick a target dt randomly between min and max
        target_dt = random.randint(min_dt_ms, max_dt_ms)
        t_i = rows[i]["timestamp_ms"]
        
        # Binary search or just scan forward to find j
        j_pos = i_pos + 1
        found_j = False
        while j_pos < len(valid_idx):
            j = valid_idx[j_pos]
            dt = rows[j]["timestamp_ms"] - t_i
            if dt >= target_dt:
                found_j = True
                break
            j_pos += 1
            
        if found_j:
            dt = rows[j]["timestamp_ms"] - t_i
            if min_dt_ms <= dt <= max_dt_ms * 1.5:  # giving slight tolerance
                return i, j

        attempts += 1

    return valid_idx[0], valid_idx[-1]


def fill_template(
    template: Dict[str, Any],
    rows: List[Dict[str, Any]],
    start_idx: int,
    end_idx: int,
    eps_1: float,
    eps_2: float,
    eps_3: float,
    delta_1: int,
) -> Optional[Dict[str, Any]]:
    template_type = template["type"]
    tmpl_text: str = template["template"]
    answer_format: Dict[str, Any] = template["answer_format"]
    
    t1 = float(rows[start_idx]["timestamp_ms"])
    t2 = float(rows[end_idx]["timestamp_ms"])
    t_ms = t1 
    
    # Dynamic joint detection
    num_joints = get_num_joints(rows)
    if num_joints == 0:
        return None
        
    axis = random.randint(0, num_joints - 1)
    axis_label = random.choice(["x", "y", "z"])
    
    # Threshold sampling
    accel_threshold = round(random.uniform(0.5, 2.0), 1)
    jerk_threshold = round(random.uniform(1.0, 10.0), 1)
    torque_threshold = round(random.uniform(0.5, 5.0), 1)
    tracking_error_threshold = round(random.uniform(0.005, 0.05), 3)
    speed_threshold = round(random.uniform(0.05, 0.5), 2)
    current_threshold = round(random.uniform(0.1, 1.5), 1)

    # Machine metadata for semantic templates
    # We'll try to guess machine_id based on dataset name
    # "inter_aursad" -> UR3e (id 0)
    # "inter_vorausad" -> UR5 (id 2)
    # For Level 1, these are the primary ones.
    # We could also look at number of joints.
    machine_id = 0 # Default to UR3e
    if num_joints == 6:
        # Check dataset stem if possible, otherwise stick to 0/2
        pass
        
    machine = get_machine_by_id(machine_id)

    options = {}
    
    if template_type == "state_joint_moved":
        truth = answer_q1_state_joint_moved(rows, t1, t2, axis, eps_1)
        if truth["answer"] in ["Unknown", "N/A", "Error"] or (truth["answer"] == "D" and not truth.get("is_true", True)):
            return None
        answer = truth["answer"]
        options = truth.get("options", {})
        question = tmpl_text.format(axis=axis, t1=t1, t2=t2, eps_1=eps_1)
        
    elif template_type == "state_friction_increase":
        truth = answer_q2_state_friction_increase(rows, t1, t2, axis, eps_2, delta_1)
        if truth["answer"] in ["Unknown", "N/A", "Error"] or (truth["answer"] == "D" and not truth.get("is_true", True)):
            return None
        answer = truth["answer"]
        options = truth.get("options", {})
        question = tmpl_text.format(axis=axis, t1=t1, t2=t2, eps_2=eps_2, delta_1=delta_1)

    elif template_type == "state_acceleration":
        truth = answer_q3_state_acceleration(rows, t_ms)
        if truth["answer"] in ["Unknown", "N/A", "Error"]: return None
        answer = truth["answer"]
        question = tmpl_text.format(t_ms=t_ms)

    elif template_type == "state_external_force_detected":
        truth = answer_q4_state_external_force_detected(rows, t_ms, eps_3)
        if truth["answer"] in ["Unknown", "N/A", "Error"] or (truth["answer"] == "D" and not truth.get("is_true", True)):
            return None
        answer = truth["answer"]
        options = truth.get("options", {})
        question = tmpl_text.format(t_ms=t_ms, eps_3=eps_3)

    elif template_type == "state_signal_statistic":
        signal_choice = random.choice(["effort_current", "feedback_speed", "feedback_pos"])
        statistic_choice = random.choice(["mean", "max", "min"])
        truth = answer_q5_state_signal_statistic(rows, t1, t2, axis, signal_choice, statistic_choice)
        if truth["answer"] in ["Unknown", "N/A", "Error"]: return None
        answer = truth["answer"]
        question = tmpl_text.format(
            axis=axis, 
            t1=t1, 
            t2=t2, 
            signal_description=signal_choice.replace("_", " "), 
            statistic_type=statistic_choice, 
            units="A" if "current" in signal_choice else ("rad/s" if "speed" in signal_choice else "rad"),
            t_ms=t_ms
        )

    elif template_type == "state_joint_speed_ranking":
        total_joints = get_num_joints(rows)
        joints_list = random.sample(range(total_joints), min(4, total_joints))
        while len(joints_list) < 4:
            joints_list.append(joints_list[-1])
            
        truth = answer_q6_state_joint_speed_ranking(rows, t_ms, joints_list)
        if truth["answer"] in ["Unknown", "N/A", "Error"]: return None
        answer = truth["answer"]
        options = truth.get("options", {})
        joints_list_str = ", ".join(map(str, joints_list))
        question = tmpl_text.format(t_ms=t_ms, joints_list=joints_list_str)

    elif template_type == "state_joint_within_rated_speed":
        total_joints = get_num_joints(rows)
        joints_list = random.sample(range(total_joints), min(4, total_joints))
        while len(joints_list) < 4:
            joints_list.append(joints_list[-1])
        
        truth = answer_q7_state_joint_within_rated_speed(rows, t_ms, machine_id, joints_list)
        if truth["answer"] in ["Unknown", "N/A", "Error"]: return None
        
        answer = truth["answer"]
        options = truth.get("options", {})
        question = tmpl_text.format(
            t_ms=t_ms,
            axis_a=joints_list[0],
            axis_b=joints_list[1],
            axis_c=joints_list[2],
            axis_d=joints_list[3]
        )

    elif template_type == "state_current_within_rated":
        truth = answer_q8_state_current_within_rated(rows, t_ms, axis, machine_id)
        if truth["answer"] in ["Unknown", "N/A", "Error"] or (truth["answer"] == "D" and not truth.get("is_true", True)):
            return None
        answer = truth["answer"]
        options = truth.get("options", {})
        question = tmpl_text.format(axis=axis, t_ms=t_ms)

    elif template_type == "state_signal_description":
        signal_choice = random.choice(["effort_current", "feedback_speed", "feedback_pos"])
        truth = answer_q9_state_signal_description(rows, t1, t2, axis, signal_choice)
        if truth["answer"] in ["Unknown", "N/A", "Error"] or (truth["answer"] == "D" and not truth.get("is_true", True)):
            return None

        answer = truth["answer"]
        options = truth.get("options", {})
        question = tmpl_text.format(
            signal=signal_choice.replace("_", " "),
            axis=axis,
            t1=t1,
            t2=t2
        )

    elif template_type == "state_safety_mode":
        truth = answer_q10_state_safety_mode(rows, t_ms, machine_id)
        if truth["answer"] in ["Unknown", "N/A", "Error"]: return None
        answer = truth["answer"]
        options = truth.get("options", {})
        question = tmpl_text.format(t_ms=t_ms)

    else:
        logger.warning(f"Unknown template type: {template_type}")
        return None

    return {
        "question": question,
        "answer_format": answer_format,
        "options": options,
        "answer": answer,
        "reasoning": truth["reasoning"],
        "anchor_timestamps": truth.get("anchor_timestamps"),
        "acceptance_bounds": truth.get("acceptance_bounds"),
        "important_features": truth.get("important_features")
    }



def generate_level1_questions(
    output_dir: Path,
    templates: List[Dict[str, Any]],
    n: int = 100,
    seed: Optional[int] = None,
    min_dt_ms: int = 100,
    max_dt_ms: int = 2000,
    eps_q: float = 1e-3,
    eps_1: float = None,
    eps_2: float = None,
    delta_1: int = None,
    eps_3: float = None,
    dataset_repo: str = "Forgis/FactoryNet_Dataset",
    test_mode: bool = False,
    questions_per_template: Optional[int] = None,
) -> None:
    if seed is not None:
        random.seed(seed)

    if eps_1 is None: eps_1 = eps_q
    if eps_2 is None: eps_2 = eps_q
    if delta_1 is None: delta_1 = max_dt_ms // 2
    if eps_3 is None: eps_3 = eps_q

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Discovering normalized episodes from Hugging Face dataset: {dataset_repo}")
    all_files = list_repo_files(dataset_repo, repo_type="dataset")
    valid_files = [f for f in all_files if f.endswith(".parquet") and ("data/normalized" in f or "data/raw" in f)]

    episodes_by_dataset = {ds: [] for ds in VALID_DATASETS}
    for f in valid_files:
        stem = Path(f).stem
        for ds in VALID_DATASETS:
            if stem.startswith(ds + "_") or stem == ds or stem.startswith(ds): # Intercept naming
                episodes_by_dataset[ds].append(f)
                break

    available_datasets = [ds for ds, files in episodes_by_dataset.items() if len(files) > 0]
    
    if not available_datasets:
        raise FileNotFoundError(f"No usable datasets found in huggingface repo: {dataset_repo}")
        
    episode_cache: Dict[str, List[Dict[str, Any]]] = {}

    def load_episode(path: str) -> List[List[Dict[str, Any]]]:
        if path not in episode_cache:
            # Use hf_hub_download which caches files locally after first download
            logger.info(f"Loading episode (cached): {path}")
            local_path = hf_hub_download(
                repo_id=dataset_repo, repo_type="dataset", filename=path
            )

            # If in test mode, only read the first 5000 rows to drastically speed up execution
            import pyarrow.parquet as pq
            if test_mode:
                logger.info("TEST MODE EXTRACTING 5000 ROWS ONLY")
                df = pd.read_parquet(local_path)
                df = df.head(5000)
            else:
                df = pd.read_parquet(local_path)
                
            if "time_s" in df.columns and "timestamp_ms" not in df.columns:
                df["timestamp_ms"] = df["time_s"] * 1000.0
            
            # Split into monotonic segments based on negative jumps in timestamp_ms
            # This is crucial because some parquet files contain independent segments
            # that reset time, which breaks binary search in mc_truth.
            records = df.to_dict(orient="records")
            segments = []
            current_segment = []
            last_ts = -float('inf')
            
            for row in records:
                ts = row.get("timestamp_ms", 0)
                if ts < last_ts:
                    if len(current_segment) >= 10:
                        segments.append(current_segment)
                    current_segment = []
                current_segment.append(row)
                last_ts = ts
            
            if len(current_segment) >= 10:
                segments.append(current_segment)
            
            if not segments:
                segments = [records] # Fallback
                
            episode_cache[path] = segments
        return episode_cache[path]

    generated = 0

    # Track how many we've generated for each template ID
    template_counts = {t["id"]: 0 for t in templates}

    if questions_per_template is not None:
        target_total = len(templates) * questions_per_template
    else:
        target_total = n

    attempts = 0
    max_attempts = target_total * 50

    while generated < target_total and attempts < max_attempts:
        attempts += 1

        ds = random.choice(available_datasets)
        ep_path = random.choice(episodes_by_dataset[ds])
        segments = load_episode(ep_path)
        
        if not segments:
            continue
            
        rows = random.choice(segments)
        
        if len(rows) < 10:
            continue

        try:
            start_idx, end_idx = pick_time_window(rows, min_dt_ms, max_dt_ms)
        except ValueError:
            continue

        if questions_per_template is not None:
            # Filter templates to only those that haven't reached their per-template quota
            valid_templates = [t for t in templates if template_counts[t["id"]] < questions_per_template]
            if not valid_templates:
                break
        else:
            valid_templates = templates
            
        template = random.choice(valid_templates)
        
        filled = fill_template(
            template, rows, start_idx, end_idx,
            eps_1=eps_1, eps_2=eps_2, eps_3=eps_3, delta_1=delta_1
        )
        if filled is None:
            continue

        # Extract context around the relevant window
        margin = max(10, (end_idx - start_idx) // 2)
        ctx_start = max(0, start_idx - margin)
        ctx_end = min(len(rows), end_idx + margin)
        subseries = rows[ctx_start:ctx_end]
        context = build_context(
            subseries, 
            template_type=template["type"],
            anchor_timestamps=filled.get("anchor_timestamps"),
            important_features=filled.get("important_features")
        )

        item = {
            "id": str(uuid.uuid4()),
            "level": 1,
            "template_id": template["id"],
            "template_type": template["type"],
            "answer_format": filled["answer_format"],
            "question": filled["question"],
            "options": filled["options"],
            "answer": filled["answer"],
            "reasoning": filled["reasoning"],
            "provenance": {
                "dataset": ds,
                "episode": Path(ep_path).stem,
                "time_window": [float(rows[start_idx]["timestamp_ms"]), float(rows[end_idx]["timestamp_ms"])]
            },
            "context": context,
        }

        out_path = output_dir / f"level1_{generated:04d}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)

        logger.info(f"✓ [{generated + 1}/{target_total}] {out_path.name} (template {template['id']}, {ds})")
        generated += 1
        template_counts[template["id"]] += 1

    if generated < target_total:
        logger.warning(f"Only generated {generated}/{target_total} questions.")
    else:
        logger.info(f"Done: {generated} questions written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Level 1 Q&A pairs directly from Hugging Face.")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/questions/level1"), help="Output directory")
    parser.add_argument("-n", type=int, default=100, help="Number of questions to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--min-dt-ms", type=int, default=100)
    parser.add_argument("--max-dt-ms", type=int, default=2000)
    parser.add_argument("--eps-q", type=float, default=1e-3)
    parser.add_argument("--templates", type=Path, default=Path("src/question_generation/level1/question_template.json"))
    parser.add_argument("--dataset-repo", type=str, default="Forgis/FactoryNet_Dataset", help="Hugging Face Dataset Repo")
    parser.add_argument("--test-mode", action="store_true", help="Only download a lightweight fraction of an episode for testing")
    parser.add_argument("--questions-per-template", type=int, default=None, help="Generate exactly X questions per template (overrides -n)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    templates = load_templates(args.templates)

    generate_level1_questions(
        output_dir=args.output_dir,
        templates=templates,
        n=args.n,
        seed=args.seed,
        min_dt_ms=args.min_dt_ms,
        max_dt_ms=args.max_dt_ms,
        eps_q=args.eps_q,
        dataset_repo=args.dataset_repo,
        test_mode=args.test_mode,
        questions_per_template=args.questions_per_template,
    )

if __name__ == "__main__":
    main()
