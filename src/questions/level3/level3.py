"""
Level 3 question generator: Root Cause Analysis.

Reads normalized episode JSON files, samples random sub time series (length 32-128),
maps fault_label to a root cause and anomaly, and writes question items to:
  datasets/questions/level3/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def load_phrases(phrases_path: Path) -> List[str]:
	if not phrases_path.exists():
		return ["Determine the root cause of the anomaly in this time series."]
	data = load_json(phrases_path)
	phrases: List[str] = []
	if isinstance(data, dict) and "mcq_templates" in data:
		for item in data["mcq_templates"]:
			text = item.get("question_text")
			if isinstance(text, str):
				phrases.append(text)
	if not phrases:
		return ["Determine the root cause of the anomaly in this time series."]
	return phrases


def load_root_causes(path: Path) -> Dict[int, Dict[str, Any]]:
	causes = load_json(path)
	by_id: Dict[int, Dict[str, Any]] = {}
	for item in causes:
		fid = item.get("fault_id")
		if isinstance(fid, int):
			by_id[fid] = item
	return by_id


def load_anomalies(path: Path) -> Dict[str, Dict[str, Any]]:
	anomalies = load_json(path)
	by_name: Dict[str, Dict[str, Any]] = {}
	for item in anomalies:
		name = item.get("anomaly_name")
		if isinstance(name, str):
			by_name[name] = item
	return by_name


def load_machines(path: Path) -> Dict[int, Dict[str, Any]]:
	machines = load_json(path)
	by_id: Dict[int, Dict[str, Any]] = {}
	if isinstance(machines, list):
		for item in machines:
			mid = item.get("machine_id")
			if isinstance(mid, int):
				by_id[mid] = item
	return by_id


def pick_fault_label(rows: List[Dict[str, Any]]) -> int:
	labels: List[int] = []
	for row in rows:
		val = row.get("fault_label")
		if val is None:
			continue
		try:
			labels.append(int(val))
		except (TypeError, ValueError):
			continue
	if not labels:
		return 0
	return max(set(labels), key=labels.count)


def sample_subseries(rows: List[Dict[str, Any]], min_len: int, max_len: int) -> List[Dict[str, Any]]:
	if not rows:
		return []
	length = len(rows)
	size = random.randint(min_len, min(max_len, length))
	if size <= 0:
		return rows
	start = random.randint(0, length - size)
	subseries = rows[start : start + size]
	
	# Normalize timestamp_ms to start at 0
	if subseries and "timestamp_ms" in subseries[0]:
		first_ts = subseries[0].get("timestamp_ms")
		if first_ts is not None:
			try:
				first_ts = float(first_ts)
				subseries = [
					{**row, "timestamp_ms": float(row.get("timestamp_ms", 0)) - first_ts if row.get("timestamp_ms") is not None else None}
					for row in subseries
				]
			except (TypeError, ValueError):
				pass
	
	return subseries


def strip_null_features(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	cleaned: List[Dict[str, Any]] = []
	for row in rows:
		cleaned.append({k: v for k, v in row.items() if v is not None})
	return cleaned


def remove_fault_id(root_cause: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
	if root_cause is None:
		return None
	return {k: v for k, v in root_cause.items() if k != "fault_id"}


def generate_level3_questions(
	input_dir: Path,
	output_dir: Path,
	phrases_path: Path,
	root_causes_path: Path,
	anomalies_path: Path,
	machines_path: Path,
	min_len: int = 32,
	max_len: int = 64,
	samples_per_episode: int = 1,
	seed: Optional[int] = None,
) -> None:
	if seed is not None:
		random.seed(seed)

	phrases = load_phrases(phrases_path)
	root_causes = load_root_causes(root_causes_path)
	anomalies = load_anomalies(anomalies_path)
	machines = load_machines(machines_path)

	output_dir.mkdir(parents=True, exist_ok=True)

	episode_files = sorted(input_dir.glob("*.json"))
	if not episode_files:
		raise FileNotFoundError(f"No episode JSON files found in {input_dir}")

	for episode_path in episode_files:
		rows = load_json(episode_path)
		if not isinstance(rows, list):
			logger.warning(f"Skipping non-list episode: {episode_path}")
			continue

		# Load metadata if it exists
		metadata_path = episode_path.parent / f"{episode_path.stem}_metadata.json"
		metadata = None
		if metadata_path.exists():
			metadata = load_json(metadata_path)

		# Extract experiment number from filename (e.g., "experiment_1.json" -> "1")
		episode_stem = episode_path.stem
		if episode_stem.startswith("experiment_"):
			exp_id = episode_stem.replace("experiment_", "")
		else:
			exp_id = episode_stem

		for sample_idx in range(samples_per_episode):
			subseries = sample_subseries(rows, min_len, max_len)
			fault_label = pick_fault_label(subseries)
			root_cause = root_causes.get(fault_label)
			root_cause = remove_fault_id(root_cause)

			anomalies_list = []
			if root_cause:
				possible = root_cause.get("possible_anomalies", [])
				for anomaly_name in possible:
					anomaly_obj = anomalies.get(anomaly_name)
					if anomaly_obj:
						anomalies_list.append(anomaly_obj)

			question = random.choice(phrases)

			out_file = output_dir / f"experiment_{exp_id}_prompt_{sample_idx}.json"
			
			# Get the machine object corresponding to the machine_id from metadata
			machine_obj = None
			if metadata:
				machine_id = metadata.get("machine_id")
				if isinstance(machine_id, int) and machine_id in machines:
					machine_obj = machines[machine_id]
			
			item = {
				"question": question,
				"time_series": strip_null_features(subseries),
				"root_cause": root_cause,
				"possible_anomalies": anomalies_list,
				"source_episode": episode_path.name,
				"metadata": metadata,
				"machine": machine_obj,
			}
			with out_file.open("w", encoding="utf-8") as f:
				json.dump(item, f, indent=2)

			logger.info(f"✓ Wrote {out_file}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate Level 3 root-cause questions")
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("datasets/normalized_episodes/aursad"),
		help="Directory of normalized episode JSON files",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("datasets/questions/level3/prompts"),
		help="Output directory for generated questions",
	)
	parser.add_argument(
		"--min-len",
		type=int,
		default=32,
		help="Minimum subseries length",
	)
	parser.add_argument(
		"--max-len",
		type=int,
		default=64,
		help="Maximum subseries length",
	)
	parser.add_argument(
		"--samples-per-episode",
		type=int,
		default=1,
		help="Number of questions to sample per episode",
	)
	parser.add_argument("--seed", type=int, default=None, help="Random seed")
	parser.add_argument("-v", "--verbose", action="store_true")

	args = parser.parse_args()
	logging.basicConfig(
		level=logging.DEBUG if args.verbose else logging.INFO,
		format="%(levelname)s: %(message)s",
	)

	repo_root = Path(__file__).resolve().parents[3]
	phrases_path = Path(__file__).with_name("phrases_level3.json")
	root_causes_path = repo_root / "datasets" / "rca" / "root_causes.json"
	anomalies_path = repo_root / "datasets" / "rca" / "anomalies.json"
	machines_path = repo_root / "datasets" / "machines" / "machines.json"

	generate_level3_questions(
		input_dir=args.input,
		output_dir=args.output,
		phrases_path=phrases_path,
		root_causes_path=root_causes_path,
		anomalies_path=anomalies_path,
		machines_path=machines_path,
		min_len=args.min_len,
		max_len=args.max_len,
		samples_per_episode=args.samples_per_episode,
		seed=args.seed,
	)


if __name__ == "__main__":
	main()
