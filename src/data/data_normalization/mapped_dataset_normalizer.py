"""
Normalize datasets to UR3e schema using feature-mapping JSON files.

Usage:
  python -m src.data.data_normalization.mapped_dataset_normalizer \
    --dataset aursad \
    --input datasets/open_datasets/aursad/AURSAD.csv \
    --output datasets/normalized_episodes \
    --episode-column episode_id \
    --max-episodes 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


AXES = list(range(6))
FORCE_TORQUE = list(range(6))
VIB = list(range(3))


def expand_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    expanded: Dict[str, str] = {}
    for dest_key, src_key in mapping.items():
        if "{axis}" in dest_key or "{axis}" in src_key:
            for axis in AXES:
                expanded[dest_key.replace("{axis}", str(axis))] = src_key.replace("{axis}", str(axis))
            continue
        if "{i}" in dest_key or "{i}" in src_key:
            for i in FORCE_TORQUE:
                expanded[dest_key.replace("{i}", str(i))] = src_key.replace("{i}", str(i))
            continue
        if "{k}" in dest_key or "{k}" in src_key:
            for k in VIB:
                expanded[dest_key.replace("{k}", str(k))] = src_key.replace("{k}", str(k))
            continue
        expanded[dest_key] = src_key
    return expanded


def load_mapping(dataset_name: str, repo_root: Path) -> Tuple[Dict[str, str], List[str], Dict[str, Any]]:
    mapping_path = repo_root / "datasets" / "mappings_of_features" / f"{dataset_name}.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    mapping = expand_mapping(config.get("mapping", {}))
    absent = config.get("absent", [])
    faults = config.get("faults", {})
    return mapping, absent, faults


def find_input_csvs(dataset_name: str, repo_root: Path, input_path: Optional[Path]) -> List[Path]:
    if input_path:
        if input_path.is_dir():
            csv_files = sorted(input_path.glob("experiment_*.csv"))
            if not csv_files:
                csv_files = sorted(input_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {input_path}")
            return csv_files
        return [input_path]

    dataset_dir = repo_root / "datasets" / "open_datasets" / dataset_name
    csv_files = sorted(dataset_dir.glob("experiment_*.csv"))
    if not csv_files:
        csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
    return csv_files


def build_schema_fields(mapping: Dict[str, str], absent: List[str]) -> List[str]:
    fields = set(mapping.keys()) | set(absent)
    return sorted(fields)


def remove_null_features(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Remove any feature (key) that is null in all rows."""
	if not rows:
		return rows
	
	# Identify features that have at least one non-null value
	features_with_values: set = set()
	for row in rows:
		for key, value in row.items():
			if value is not None:
				features_with_values.add(key)
	
	# Filter rows to keep only features with at least one non-null value
	cleaned_rows: List[Dict[str, Any]] = []
	for row in rows:
		cleaned_rows.append({k: v for k, v in row.items() if k in features_with_values})
	
	return cleaned_rows


def normalize_fault_id(value: Any) -> Any:
    if value is None:
        return 0
    if isinstance(value, str) and not value.strip():
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def map_fault_label(value: Any, faults: Dict[str, Any]) -> Any:
    if value is None or not faults:
        return value
    key = str(value)
    if key in faults:
        return faults[key]
    return value


def build_row_dict(
    row: pd.Series,
    mapping: Dict[str, str],
    schema_fields: List[str],
    faults: Dict[str, Any],
) -> Dict[str, Any]:
    row_dict: Dict[str, Any] = {}
    for out_field in schema_fields:
        if out_field in mapping:
            src_field = mapping[out_field]
            value = row.get(src_field, None)
            if pd.isna(value):
                value = None
            if out_field == "fault_label":
                value = map_fault_label(value, faults)
            row_dict[out_field] = value
        else:
            row_dict[out_field] = None
    return row_dict


def build_episode_rows(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    absent: List[str],
    schema_fields: List[str],
    faults: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        rows.append(build_row_dict(row, mapping, schema_fields, faults))

    return rows


def write_episode(
    episode_rows: List[Dict[str, Any]],
    output_dir: Path,
    episode_id: str,
    source_file: str,
    include_metadata: bool,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows = remove_null_features(episode_rows)
    output_file = output_dir / f"{episode_id}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(episode_rows, f, indent=2)

    if include_metadata:
        first_ts = episode_rows[0].get("timestamp_ms") if episode_rows else None
        last_ts = episode_rows[-1].get("timestamp_ms") if episode_rows else None
        machine_id = episode_rows[0].get("machine_id") if episode_rows else None
        metadata = {
            "episode_id": episode_id,
            "source_file": source_file,
            "num_samples": len(episode_rows),
            "schema": "ur3e_v1",
            "first_timestamp_ms": first_ts,
            "last_timestamp_ms": last_ts,
            "duration_ms": (last_ts - first_ts) if (first_ts is not None and last_ts is not None) else None,
            "machine_id": machine_id,
        }
        metadata_file = output_dir / f"{episode_id}_metadata.json"
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return str(output_file)


def write_episode_streaming(
    episode_rows: Iterable[Dict[str, Any]],
    output_dir: Path,
    episode_id: str,
    source_file: str,
    include_metadata: bool,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{episode_id}.json"

    # Collect all rows first to identify null features
    all_rows = list(episode_rows)
    all_rows = remove_null_features(all_rows)

    num_samples = 0
    first_ts = None
    last_ts = None

    with output_file.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for row in all_rows:
            if not first:
                f.write(",\n")
            json.dump(row, f, indent=2)
            first = False
            num_samples += 1
            ts = row.get("timestamp_ms")
            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
        f.write("\n]\n")

    if include_metadata:
        machine_id = all_rows[0].get("machine_id") if all_rows else None
        metadata = {
            "episode_id": episode_id,
            "source_file": source_file,
            "num_samples": num_samples,
            "schema": "ur3e_v1",
            "first_timestamp_ms": first_ts,
            "last_timestamp_ms": last_ts,
            "duration_ms": (last_ts - first_ts) if (first_ts is not None and last_ts is not None) else None,
            "machine_id": machine_id,
        }
        metadata_file = output_dir / f"{episode_id}_metadata.json"
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return str(output_file)


def normalize_dataset(
    dataset_name: str,
    input_csv: Path,
    output_root: Path,
    episode_column: Optional[str],
    max_episodes: Optional[int],
    episode_size: int,
    include_metadata: bool,
    output_basename: Optional[str] = None,
) -> List[Dict[str, str]]:
    repo_root = Path(__file__).resolve().parents[3]
    mapping, absent, faults = load_mapping(dataset_name, repo_root)
    schema_fields = build_schema_fields(mapping, absent)

    results: List[Dict[str, str]] = []
    output_dir = output_root / dataset_name

    if episode_column:
        class EpisodeWriter:
            def __init__(self, episode_id: str) -> None:
                self.episode_id = episode_id
                self.output_file = output_dir / f"{episode_id}.json"
                self.handle = self.output_file.open("w", encoding="utf-8")
                self.handle.write("[\n")
                self.first = True
                self.num_samples = 0
                self.first_ts = None
                self.last_ts = None
                self.machine_id = None

            def write_row(self, row_dict: Dict[str, Any]) -> None:
                if not self.first:
                    self.handle.write(",\n")
                json.dump(row_dict, self.handle, indent=2)
                self.first = False
                self.num_samples += 1
                ts = row_dict.get("timestamp_ms")
                if ts is not None:
                    if self.first_ts is None:
                        self.first_ts = ts
                    self.last_ts = ts
                if self.machine_id is None:
                    self.machine_id = row_dict.get("machine_id")

            def close(self) -> None:
                self.handle.write("\n]\n")
                self.handle.close()

        writers: Dict[str, EpisodeWriter] = {}
        episode_order: List[str] = []
        stop_reading = False

        episode_prefix = output_basename or dataset_name

        for chunk in pd.read_csv(input_csv, chunksize=10000, low_memory=False):
            if episode_column not in chunk.columns:
                raise ValueError(f"Episode column '{episode_column}' not found in CSV")

            for _, row in chunk.iterrows():
                ep_value = row.get(episode_column)
                if pd.isna(ep_value):
                    ep_value = "unknown"
                ep_key = str(ep_value)

                if ep_key not in writers:
                    if max_episodes is not None and len(episode_order) >= max_episodes:
                        stop_reading = True
                        break
                    ep_id = f"{episode_prefix}_{ep_key}"
                    writers[ep_key] = EpisodeWriter(ep_id)
                    episode_order.append(ep_key)

                row_dict = build_row_dict(row, mapping, schema_fields, faults)
                writers[ep_key].write_row(row_dict)

            if stop_reading:
                break

        for ep_key in episode_order:
            writer = writers[ep_key]
            writer.close()
            json_path = str(output_dir / f"{writer.episode_id}.json")

            if include_metadata:
                metadata = {
                    "episode_id": writer.episode_id,
                    "source_file": input_csv.name,
                    "num_samples": writer.num_samples,
                    "schema": "ur3e_v1",
                    "first_timestamp_ms": writer.first_ts,
                    "last_timestamp_ms": writer.last_ts,
                    "duration_ms": (writer.last_ts - writer.first_ts)
                    if (writer.first_ts is not None and writer.last_ts is not None)
                    else None,
                    "machine_id": writer.machine_id,
                }
                metadata_file = output_dir / f"{writer.episode_id}_metadata.json"
                with metadata_file.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

            results.append({
                "dataset": dataset_name,
                "episode_id": writer.episode_id,
                "json_file": json_path,
            })
    else:
        if max_episodes is not None and max_episodes < 1:
            return results

        episode_count = 0
        row_in_episode = 0
        stop_reading = False
        all_rows: List[Dict[str, Any]] = []

        for chunk in pd.read_csv(input_csv, chunksize=10000, low_memory=False):
            for _, row in chunk.iterrows():
                if stop_reading:
                    break

                row_in_episode += 1
                row_dict = build_row_dict(row, mapping, schema_fields, faults)
                all_rows.append(row_dict)

                if row_in_episode >= episode_size:
                    episode_count += 1
                    row_in_episode = 0
                    if max_episodes is not None and episode_count >= max_episodes:
                        stop_reading = True
                        break

            if stop_reading:
                break

        output_dir.mkdir(parents=True, exist_ok=True)
        all_rows = remove_null_features(all_rows)
        output_name = output_basename or dataset_name
        output_file = output_dir / f"{output_name}.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2)

        json_path = str(output_file)
        results.append({
            "dataset": dataset_name,
            "episode_id": output_name,
            "json_file": json_path,
        })

        if include_metadata:
            first_ts = all_rows[0].get("timestamp_ms") if all_rows else None
            last_ts = all_rows[-1].get("timestamp_ms") if all_rows else None
            machine_id = all_rows[0].get("machine_id") if all_rows else None
            metadata = {
                "episode_id": output_name,
                "source_file": input_csv.name,
                "num_samples": len(all_rows),
                "num_episodes": episode_count if row_in_episode == 0 else episode_count + 1,
                "schema": "ur3e_v1",
                "first_timestamp_ms": first_ts,
                "last_timestamp_ms": last_ts,
                "duration_ms": (last_ts - first_ts)
                if (first_ts is not None and last_ts is not None)
                else None,
                "machine_id": machine_id,
            }
            metadata_file = output_dir / f"{output_name}_metadata.json"
            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize dataset using feature mapping")
    parser.add_argument("--dataset", required=True, help="Dataset name (mapping file in datasets/mappings_of_features)")
    parser.add_argument("--input", type=Path, help="Path to input CSV (defaults to datasets/open_datasets/<dataset>/*.csv)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/normalized_episodes"),
        help="Output directory for normalized JSON",
    )
    parser.add_argument(
        "--episode-column",
        type=str,
        default=None,
        help="Optional column to group rows into episodes",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="How many episodes to normalize (default: all)",
    )
    parser.add_argument(
        "--episode-size",
        type=int,
        default=1,
        help="Number of rows per episode (default: 1 = each row is an episode)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip metadata file generation",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[3]
    input_csvs = find_input_csvs(args.dataset, repo_root, args.input)

    results: List[Dict[str, str]] = []
    multi_input = len(input_csvs) > 1

    for csv_path in input_csvs:
        output_basename = None
        if multi_input or csv_path.stem.startswith("experiment_"):
            output_basename = csv_path.stem

        results.extend(
            normalize_dataset(
                dataset_name=args.dataset,
                input_csv=csv_path,
                output_root=args.output,
                episode_column=args.episode_column,
                max_episodes=args.max_episodes,
                episode_size=args.episode_size,
                include_metadata=not args.no_metadata,
                output_basename=output_basename,
            )
        )

    if not results:
        logger.warning("No episodes were normalized.")
        return 1

    logger.info(f"✓ Normalization complete! Processed {len(results)} episode(s).")
    for result in results:
        logger.info(f"  - {result['dataset']}/{result['episode_id']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
