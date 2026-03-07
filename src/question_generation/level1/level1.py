"""
Level 1 question generator: State identification.

Reads normalized episode JSON files from datasets,
samples random time windows across the entire sequence, and fills Level 1 question templates.
Answers are generated deterministically using raw values from episode readings.

Output: datasets/questions/level1/level1_{NNNN}.json

Usage:
    python -m src.question_generation.level1.level1 -n 100 --seed 42
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
from huggingface_hub import list_repo_files

from src.question_generation.utils.io import load_json, load_templates
from src.question_generation.utils.template import build_context
from src.question_generation.level1.mc_truth import (
    answer_q1_position_check,
    answer_q2_friction_increase,
    answer_q3_end_effector_accel,
    answer_q4_external_force,
    answer_q5_joint_jerk,
    answer_q6_torque_magnitude,
    answer_q7_joint_speed_ranking,
)

logger = logging.getLogger(__name__)

VALID_DATASETS = ["inter_aursad", "inter_vorausad", "aursad", "vorausad"]


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
    tid = template["id"]
    tmpl_text: str = template["template"]
    answer_format: Dict[str, Any] = template["answer_format"]
    
    t1 = float(rows[start_idx]["timestamp_ms"])
    t2 = float(rows[end_idx]["timestamp_ms"])
    t_ms = t1 
    
    axis = random.randint(0, 5)
    axis_label = random.choice(["x", "y", "z"])

    options = {}
    
    if tid == 1:
        truth = answer_q1_position_check(rows, t1, t2, axis, eps_1)
        if truth["answer"] == "Unknown": return None
        answer = truth["answer"]
        question = tmpl_text.format(axis=axis, t1=t1, t2=t2, eps_1=eps_1)
        
    elif tid == 2:
        truth = answer_q2_friction_increase(rows, t1, t2, axis, eps_2, delta_1)
        if truth["answer"] == "Unknown": return None
        answer = truth["answer"]
        question = tmpl_text.format(axis=axis, t1=t1, t2=t2, eps_2=eps_2, delta_1=delta_1)
        
    elif tid == 3:
        truth = answer_q3_end_effector_accel(rows, t_ms)
        if truth["answer"] == "Unknown": return None
        answer = truth["answer"]
        question = tmpl_text.format(t_ms=t_ms)
        
    elif tid == 4:
        truth = answer_q4_external_force(rows, t_ms, eps_3)
        if truth["answer"] == "Unknown": return None
        answer = truth["answer"]
        question = tmpl_text.format(t_ms=t_ms, eps_3=eps_3)

    elif tid == 5:
        truth = answer_q5_joint_jerk(rows, t_ms, axis)
        if truth["answer"] == "Unknown": return None
        answer = truth["answer"]
        question = tmpl_text.format(axis=axis, t_ms=t_ms)

    elif tid == 6:
        truth = answer_q6_torque_magnitude(rows, t_ms, axis_label)
        if truth["answer"] == "Unknown": return None
        answer = truth["answer"]
        question = tmpl_text.format(axis_label=axis_label, t_ms=t_ms)

    elif tid == 7:
        joints = list(range(6))
        random.shuffle(joints)
        chosen_joints = joints[:4] 
        
        truth = answer_q7_joint_speed_ranking(rows, t_ms, chosen_joints)
        if truth["answer"] == "Unknown": return None
        answer = truth["answer"]
        
        letters = ["A", "B", "C", "D"]
        joints_list_str = ", ".join([f"{letters[i]}: Joint {chosen_joints[i]}" for i in range(4)])
        question = tmpl_text.format(joints_list=joints_list_str, t_ms=t_ms)
        
    else:
        logger.warning(f"Unknown template id: {tid}")
        return None

    return {
        "question": question,
        "answer_format": answer_format,
        "options": options,
        "answer": answer,
        "reasoning": truth["reasoning"]
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
    valid_files = [f for f in all_files if f.endswith(".parquet") and "data/normalized" in f]

    episodes_by_dataset = {ds: [] for ds in VALID_DATASETS}
    for f in valid_files:
        stem = Path(f).stem
        for ds in VALID_DATASETS:
            if stem.startswith(ds + "_") or stem == ds or stem.startswith(ds): # Intercept naming
                episodes_by_dataset[ds].append(f"hf://datasets/{dataset_repo}/{f}")
                break

    available_datasets = [ds for ds, files in episodes_by_dataset.items() if len(files) > 0]
    
    if not available_datasets:
        raise FileNotFoundError(f"No usable datasets found in huggingface repo: {dataset_repo}")
        
    episode_cache: Dict[str, List[Dict[str, Any]]] = {}

    def load_episode(path: str) -> List[Dict[str, Any]]:
        if path not in episode_cache:
            logger.info(f"Downloading/Reading from Hugging Face: {path}")
            
            # If in test mode, only read the first 1000 rows to drastically speed up execution
            import pyarrow.parquet as pq
            if test_mode:
                logger.info("TEST MODE EXTRACTING 1000 ROWS ONLY")
                import fsspec
                # We use pandas natively with backend filters
                df = pd.read_parquet(
                    path,
                    engine="pyarrow", 
                    filters=None,
                    storage_options=None
                )
                df = df.head(1000)
            else:
                df = pd.read_parquet(path)
                
            if "time_s" in df.columns and "timestamp_ms" not in df.columns:
                df["timestamp_ms"] = df["time_s"] * 1000.0
            episode_cache[path] = df.to_dict(orient="records")
        return episode_cache[path]

    generated = 0
    attempts = 0
    max_attempts = n * 50

    while generated < n and attempts < max_attempts:
        attempts += 1

        ds = random.choice(available_datasets)
        ep_path = random.choice(episodes_by_dataset[ds])
        rows = load_episode(ep_path)
        
        if not isinstance(rows, list) or len(rows) < 10:
            continue

        try:
            start_idx, end_idx = pick_time_window(rows, min_dt_ms, max_dt_ms)
        except ValueError:
            continue

        template = random.choice(templates)
        
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
        context = build_context(subseries)

        item = {
            "id": str(uuid.uuid4()),
            "level": 1,
            "template_id": template["id"],
            "template_type": template["type"],
            "question": filled["question"],
            "options": filled["options"],
            "answer": filled["answer"],
            "reasoning": filled["reasoning"],
            "provenance": {
                "dataset": ds,
                "episode": ep_path.split("/")[-1].replace(".parquet", ""),
                "time_window": [float(rows[start_idx]["timestamp_ms"]), float(rows[end_idx]["timestamp_ms"])]
            },
            "context": context,
        }

        out_path = output_dir / f"level1_{generated:04d}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)

        logger.info(f"✓ [{generated + 1}/{n}] {out_path.name} (template {template['id']}, {ds})")
        generated += 1

    if generated < n:
        logger.warning(f"Only generated {generated}/{n} questions.")
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
    )

if __name__ == "__main__":
    main()
