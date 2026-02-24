"""
Level 3 question generator: Root Cause Analysis.

Reads normalized episode JSON files, samples random sub time series (length 32-128),
maps fault_label to a root cause and anomaly, and writes question items to:
  datasets/questions/level3/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

INACTIVE_CONSTANT_THRESHOLD = 55
MAX_RESAMPLE_ATTEMPTS = 10
INACTIVITY_TRIM = 5


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


def remove_feature(rows: List[Dict[str, Any]], feature: str) -> List[Dict[str, Any]]:
	cleaned: List[Dict[str, Any]] = []
	for row in rows:
		cleaned.append({k: v for k, v in row.items() if k != feature})
	return cleaned


def sort_feature_keys(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	sorted_rows: List[Dict[str, Any]] = []
	for row in rows:
		sorted_rows.append({k: row[k] for k in sorted(row.keys())})
	return sorted_rows


def _count_constant_feature_keys(constant_features: Dict[str, Any]) -> int:
	count = 0
	for key in constant_features.keys():
		if key == "joint_modes":
			count += 4
		else:
			count += 1
	return count


def _trim_for_inactivity(rows: List[Dict[str, Any]], trim: int = INACTIVITY_TRIM) -> List[Dict[str, Any]]:
	if len(rows) <= trim * 2:
		return rows
	return rows[trim:-trim]


def _constant_features_for_inactivity(
	rows: List[Dict[str, Any]],
	metadata: Optional[Dict[str, Any]],
	machines: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
	# Get the machine object and extract mode enums for expansion
	machine_obj = None
	safety_modes_enum = None
	joint_modes_enum = None
	robot_modes_enum = None
	if metadata:
		machine_id = metadata.get("machine_id")
		if isinstance(machine_id, int) and machine_id in machines:
			raw_machine = machines[machine_id]
			safety_modes_enum = raw_machine.get("safety_modes", {}).get("enum")
			joint_modes_enum = raw_machine.get("joint_modes", {}).get("enum")
			robot_modes_enum = raw_machine.get("robot_modes", {}).get("enum")
			machine_obj = {k: v for k, v in raw_machine.items()
			               if k not in ["safety_modes", "joint_modes", "robot_modes"]}

	full_time_series = strip_null_features(rows)
	full_time_series = sort_feature_keys(full_time_series)
	full_time_series = expand_mode_feature(full_time_series, "safety_mode", safety_modes_enum)
	for i in range(6):
		full_time_series = expand_mode_feature(full_time_series, f"joint_mode_{i}", joint_modes_enum)
	full_time_series = expand_mode_feature(full_time_series, "robot_mode", robot_modes_enum)

	time_series = remove_feature(full_time_series, "fault_label")
	_, constant_features = remove_constant_features(time_series)
	constant_features = expand_constant_mode_feature(constant_features, "safety_mode", safety_modes_enum)
	for i in range(6):
		constant_features = expand_constant_mode_feature(constant_features, f"joint_mode_{i}", joint_modes_enum)
	constant_features = expand_constant_mode_feature(constant_features, "robot_mode", robot_modes_enum)
	constant_features = consolidate_joint_modes(constant_features)

	return constant_features


def is_inactive_subseries(
	rows: List[Dict[str, Any]],
	metadata: Optional[Dict[str, Any]],
	machines: Dict[int, Dict[str, Any]],
	threshold: int = INACTIVE_CONSTANT_THRESHOLD,
) -> bool:
	trimmed = _trim_for_inactivity(rows)
	constant_features = _constant_features_for_inactivity(trimmed, metadata, machines)
	if not constant_features:
		return False
	return _count_constant_feature_keys(constant_features) >= threshold


def format_note_value(value: Any) -> Any:
	if isinstance(value, (int, float, np.floating)):
		rounded = round(float(value), 2)
		if rounded.is_integer():
			return int(rounded)
		return rounded
	return value


def expand_mode_feature(
	rows: List[Dict[str, Any]],
	feature_name: str,
	modes_enum: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
	if not modes_enum or not rows or feature_name not in rows[0]:
		return rows
	
	# Build lookup from value to full object
	lookup: Dict[Any, Dict[str, Any]] = {}
	for mode in modes_enum:
		value = mode.get("value")
		if value is not None:
			lookup[value] = mode
	
	expanded: List[Dict[str, Any]] = []
	for row in rows:
		new_row = {**row}
		mode_val = row.get(feature_name)
		if mode_val is not None and mode_val in lookup:
			new_row[feature_name] = lookup[mode_val]
		expanded.append(new_row)
	return expanded


def expand_constant_mode_feature(
	constant_features: Dict[str, Any],
	feature_name: str,
	modes_enum: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
	if not modes_enum or feature_name not in constant_features:
		return constant_features
	
	mode_val = constant_features.get(feature_name)
	# Skip if already expanded (is a dict)
	if isinstance(mode_val, dict):
		return constant_features
	
	# Build lookup from value to full object
	lookup: Dict[Any, Dict[str, Any]] = {}
	for mode in modes_enum:
		value = mode.get("value")
		if value is not None:
			lookup[value] = mode
	
	if mode_val is not None and mode_val in lookup:
		return {**constant_features, feature_name: lookup[mode_val]}
	return constant_features


def consolidate_joint_modes(constant_features: Dict[str, Any]) -> Dict[str, Any]:
	"""If all 6 joint modes have the same constant value, consolidate to single joint_modes key."""
	joint_mode_keys = [f"joint_mode_{i}" for i in range(6)]
	
	# Check if all 6 joint modes are present in constants
	if not all(key in constant_features for key in joint_mode_keys):
		return constant_features
	
	# Get all joint mode values
	joint_values = [constant_features[key] for key in joint_mode_keys]
	
	# Check if all are equal (works for both int and dict)
	first_value = joint_values[0]
	if all(v == first_value for v in joint_values[1:]):
		# All same - consolidate
		result = {k: v for k, v in constant_features.items() if k not in joint_mode_keys}
		result["joint_modes"] = first_value
		return result
	
	return constant_features


def remove_constant_features(
	rows: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
	if not rows:
		return rows, {}

	constants: Dict[str, Any] = {}
	keys = set(rows[0].keys())
	for key in keys:
		values: List[Any] = []
		for row in rows:
			value = row.get(key)
			if value is None:
				continue
			values.append(value)

		if not values:
			continue

		# Numeric features: treat small relative variation as constant
		if all(isinstance(v, (int, float, np.floating)) for v in values):
			series = np.array([float(v) for v in values], dtype=float)
			if series.size < 2:
				constants[key] = float(series.mean())
				continue
			min_val = float(series.min())
			max_val = float(series.max())
			mean_val = float(series.mean())
			if math.isclose(max_val, min_val):
				constants[key] = mean_val
				continue
			std_val = float(series.std())
			eps = 1e-9
			rel_range = (max_val - min_val) / (abs(mean_val) + eps)
			cv = std_val / (abs(mean_val) + eps)
			if rel_range < 0.02 or cv < 0.01:
				constants[key] = mean_val
			continue

		# Non-numeric features: treat exact-constant values as constant
		first_value = values[0]
		if all(value == first_value for value in values[1:]):
			constants[key] = first_value

	if not constants:
		return rows, {}

	filtered: List[Dict[str, Any]] = []
	for row in rows:
		filtered.append({k: v for k, v in row.items() if k not in constants})
	return filtered, constants


def remove_fault_id(root_cause: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
	if root_cause is None:
		return None
	return {k: v for k, v in root_cause.items() if k != "fault_id"}


def generate_prompt(
	subseries: List[Dict[str, Any]],
	metadata: Optional[Dict[str, Any]],
	question_text: str,
	root_causes: Dict[int, Dict[str, Any]],
	anomalies: Dict[str, Dict[str, Any]],
	machines: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
	"""
	Generate a single prompt dict from a subseries of time series data.
	
	Args:
		subseries: List of time series data points
		metadata: Optional metadata containing machine_id
		question_text: The question text to include in the prompt
		root_causes: Dictionary mapping fault_id to root cause objects
		anomalies: Dictionary mapping anomaly_name to anomaly objects
		machines: Dictionary mapping machine_id to machine objects
	
	Returns:
		Dictionary containing the complete prompt structure with:
		- question
		- root_cause (with embedded possible_anomalies)
		- machine
		- notes (if there are constant features)
		- time_series
	"""
	fault_label = pick_fault_label(subseries)
	root_cause = root_causes.get(fault_label)
	root_cause = remove_fault_id(root_cause)

	# Populate possible_anomalies in root_cause with full anomaly objects
	if root_cause:
		possible_names = root_cause.get("possible_anomalies", [])
		anomalies_list = []
		for anomaly_name in possible_names:
			anomaly_obj = anomalies.get(anomaly_name)
			if anomaly_obj:
				anomalies_list.append(anomaly_obj)
		root_cause = {**root_cause, "possible_anomalies": anomalies_list}

	# Get the machine object and extract mode enums for expansion
	machine_obj = None
	safety_modes_enum = None
	joint_modes_enum = None
	robot_modes_enum = None
	if metadata:
		machine_id = metadata.get("machine_id")
		if isinstance(machine_id, int) and machine_id in machines:
			raw_machine = machines[machine_id]
			safety_modes_enum = raw_machine.get("safety_modes", {}).get("enum")
			joint_modes_enum = raw_machine.get("joint_modes", {}).get("enum")
			robot_modes_enum = raw_machine.get("robot_modes", {}).get("enum")
			# Remove mode definitions from machine_obj for prompt
			machine_obj = {k: v for k, v in raw_machine.items() 
			               if k not in ["safety_modes", "joint_modes", "robot_modes"]}
	
	full_time_series = strip_null_features(subseries)
	full_time_series = sort_feature_keys(full_time_series)
	full_time_series = expand_mode_feature(full_time_series, "safety_mode", safety_modes_enum)
	for i in range(6):
		full_time_series = expand_mode_feature(full_time_series, f"joint_mode_{i}", joint_modes_enum)
	full_time_series = expand_mode_feature(full_time_series, "robot_mode", robot_modes_enum)
	
	time_series = remove_feature(full_time_series, "fault_label")
	time_series, constant_features = remove_constant_features(time_series)
	constant_features = expand_constant_mode_feature(constant_features, "safety_mode", safety_modes_enum)
	for i in range(6):
		constant_features = expand_constant_mode_feature(constant_features, f"joint_mode_{i}", joint_modes_enum)
	constant_features = expand_constant_mode_feature(constant_features, "robot_mode", robot_modes_enum)
	constant_features = consolidate_joint_modes(constant_features)
	time_series = sort_feature_keys(time_series)

	item = {
		"question": question_text,
		"root_cause": root_cause,
		"machine": machine_obj,
	}
	if constant_features:
		item["notes"] = {
			"disclaimer": "these features stayed constant at the following values",
			"constant_features": {
				k: format_note_value(constant_features[k])
				for k in sorted(constant_features.keys())
			},
		}
	item["time_series"] = time_series
	
	return item


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
	test_mode: bool = False,
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

	episode_data: List[Dict[str, Any]] = []
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

		episode_data.append({
			"exp_id": exp_id,
			"rows": rows,
			"metadata": metadata,
		})

	if not episode_data:
		raise FileNotFoundError(f"No valid episode JSON files found in {input_dir}")

	def _pick_other_episode(excluded: set[str]) -> Optional[Dict[str, Any]]:
		candidates = [ep for ep in episode_data if ep["exp_id"] not in excluded]
		if not candidates:
			return None
		return random.choice(candidates)

	for episode in episode_data:
		for sample_idx in range(samples_per_episode):
			current_episode = episode
			excluded_ids = {current_episode["exp_id"]}
			attempts = 0
			subseries: List[Dict[str, Any]] = []
			item: Dict[str, Any] = {}

			while True:
				attempts += 1
				rows = current_episode["rows"]
				metadata = current_episode["metadata"]
				exp_id = current_episode["exp_id"]

				subseries = sample_subseries(rows, min_len, max_len)
				question = random.choice(phrases)

				if not is_inactive_subseries(subseries, metadata, machines, INACTIVE_CONSTANT_THRESHOLD):
					# Generate the prompt using the modular function
					item = generate_prompt(
						subseries=subseries,
						metadata=metadata,
						question_text=question,
						root_causes=root_causes,
						anomalies=anomalies,
						machines=machines,
					)
					break

				if attempts >= MAX_RESAMPLE_ATTEMPTS:
					next_episode = _pick_other_episode(excluded_ids)
					if next_episode is None:
						logger.warning(
							f"All episodes inactive for prompt {sample_idx}; keeping last inactive sample"
						)
						break
					current_episode = next_episode
					excluded_ids.add(current_episode["exp_id"])
					attempts = 0

			out_file = output_dir / f"experiment_{exp_id}_prompt_{sample_idx}.json"
			with out_file.open("w", encoding="utf-8") as f:
				json.dump(item, f, indent=2)

			logger.info(f"✓ Wrote {out_file}")

			# Export CSV if test mode is enabled
			if test_mode:
				csv_file = out_file.with_suffix(".csv")
				# Reconstruct full_time_series for CSV export
				full_time_series = strip_null_features(subseries)
				full_time_series = sort_feature_keys(full_time_series)
				# Expand modes for CSV
				if metadata:
					machine_id = metadata.get("machine_id")
					if isinstance(machine_id, int) and machine_id in machines:
						raw_machine = machines[machine_id]
						safety_modes_enum = raw_machine.get("safety_modes", {}).get("enum")
						joint_modes_enum = raw_machine.get("joint_modes", {}).get("enum")
						robot_modes_enum = raw_machine.get("robot_modes", {}).get("enum")
						full_time_series = expand_mode_feature(full_time_series, "safety_mode", safety_modes_enum)
						for i in range(6):
							full_time_series = expand_mode_feature(full_time_series, f"joint_mode_{i}", joint_modes_enum)
						full_time_series = expand_mode_feature(full_time_series, "robot_mode", robot_modes_enum)
				
				csv_rows = full_time_series
				if csv_rows:
					# Get all field names and put timestamp_ms first
					all_keys = sorted(csv_rows[0].keys())
					if "timestamp_ms" in all_keys:
						all_keys.remove("timestamp_ms")
						fieldnames = ["timestamp_ms"] + all_keys
					else:
						fieldnames = all_keys
					with csv_file.open("w", encoding="utf-8", newline="") as csvf:
						writer = csv.DictWriter(csvf, fieldnames=fieldnames)
						writer.writeheader()
						writer.writerows(csv_rows)
					logger.info(f"✓ Wrote CSV {csv_file}")


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
	parser.add_argument("--test", action="store_true", help="Export time series as CSV files")
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
		test_mode=args.test,
	)


if __name__ == "__main__":
	main()
