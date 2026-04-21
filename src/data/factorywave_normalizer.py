"""
Normalize FactoryWave (Forgis/FactoryWave) dataset to UR3e schema format.

Reads ur_signals parquet files (pick-and-place and peg-in-hole episodes recorded
from a live UR3 robot) and the episode metadata table, then outputs per-episode
JSON files in the standardized schema used by the FactoryBench pipeline.

For counterfactual episodes:
  - Groups CFs by their baseline episode
  - Computes KL divergence between each CF's pre-fault segment and the baseline
  - Keeps only the CF with the lowest KL divergence (best match to baseline)
  - Outputs as {"baseline": [...], "counterfactual": [...]} like simulations

For non-counterfactual episodes (normal, fault, trajectory-opt):
  - Outputs as a flat list of rows

Input:
    data/factorywave/data/ur_signals.parquet (or ur_signals_10hz/ + ur_signals_125hz/)
    data/factorywave/data/episode.parquet

Output:
    <output>/factorywave/<episode_id>.json
    <output>/factorywave/<episode_id>_metadata.json

Usage:
    python -m src.data.factorywave_normalizer \\
        --input data/factorywave/data \\
        --output data/normalized_episodes
"""

import ast
import json
import math
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

try:
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("pandas and pyarrow are required. Install with: pip install pandas pyarrow")

from src.data._decimation import decimate_dataframe

logger = logging.getLogger(__name__)


class _NaNSafeEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        return super().iterencode(self._sanitize(o), _one_shot)

    def _sanitize(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        return obj


# FactoryWave ur_signals column -> UR3e schema column
def _build_column_mapping() -> Dict[str, Optional[str]]:
    m: Dict[str, Optional[str]] = {}

    for i in range(6):
        # INTENT — joint commands
        m[f"setpoint_pos_{i}"] = f"target_joint_{i}"
        m[f"setpoint_speed_{i}"] = f"target_joint_vel_{i}"
        m[f"setpoint_acc_{i}"] = f"target_joint_accel_{i}"

        # OUTCOME — joint feedback
        m[f"feedback_pos_{i}"] = f"joint_{i}"
        m[f"feedback_speed_{i}"] = f"joint_vel_{i}"

        # OUTCOME — effort / current
        m[f"effort_current_{i}"] = f"joint_current_{i}"
        m[f"effort_target_current_{i}"] = f"target_joint_current_{i}"
        m[f"effort_target_torque_{i}"] = None

        # OUTCOME — controller output
        m[f"control_output_{i}"] = f"joint_control_output_{i}"

        # CONTEXT — per-joint
        m[f"joint_temp_{i}"] = f"joint_temp_{i}"
        m[f"joint_mode_{i}"] = f"joint_mode_{i}"
        m[f"joint_voltage_{i}"] = None

    # INTENT — TCP commands
    for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
        m[f"setpoint_tcp_{i}"] = f"target_tcp_{axis}"
        m[f"setpoint_tcp_speed_{i}"] = f"target_tcp_speed_{axis}"

    # OUTCOME — TCP feedback
    for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
        m[f"feedback_tcp_{i}"] = f"tcp_{axis}"
        m[f"feedback_tcp_speed_{i}"] = f"tcp_speed_{axis}"

    # OUTCOME — forces/torques
    for i, axis in enumerate(["x", "y", "z"]):
        m[f"true_force_{i}"] = f"tcp_force_{axis}"
        m[f"true_force_{i + 3}"] = f"tcp_torque_{axis}"
    for i in range(6):
        m[f"est_contact_force_{i}"] = None

    # OUTCOME — vibration (tool accelerometer)
    for i, axis in enumerate(["x", "y", "z"]):
        m[f"vibration_{i}"] = f"tool_accel_{axis}"

    m["acoustic_0"] = None
    m["protective_stop_state"] = None

    # INTENT — gripper
    m["gripper_command"] = "force"  # gripper force as proxy for grip command

    # CONTEXT — system-level
    m["robot_mode"] = "robot_mode"
    m["safety_mode"] = "safety_mode"
    m["digital_input_bits"] = "digital_inputs"
    m["digital_output_bits"] = "digital_outputs"
    m["runtime_state"] = "runtime_state"
    m["main_voltage"] = "main_voltage"
    m["robot_voltage"] = "robot_voltage"
    m["robot_current"] = "robot_current"
    m["speed_scaling"] = "speed_scaling"
    m["target_speed_fraction"] = "target_speed_fraction"
    m["tool_momentum"] = "momentum"

    return m


COLUMN_MAPPING = _build_column_mapping()

# Integer-typed schema columns
_INT_COLS = {
    "joint_mode_0", "joint_mode_1", "joint_mode_2",
    "joint_mode_3", "joint_mode_4", "joint_mode_5",
    "robot_mode", "safety_mode", "runtime_state",
    "digital_input_bits", "digital_output_bits",
}

# Columns that should not be anti-alias filtered during decimation
_NON_CONTINUOUS = {
    "episode_id", "fault", "task_phase", "skill_index",
    "status", "busy", "connected", "streaming",
    "robot_mode", "safety_mode", "robot_status", "safety_status",
    "runtime_state", "digital_inputs", "digital_outputs",
    "frame_count", "operation_counter", "grip_detected",
}

TARGET_HZ = 10  # Target sampling rate for normalized output

# Continuous signal columns used for KL divergence computation
_KL_SIGNAL_COLS = [
    "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
    "joint_vel_0", "joint_vel_1", "joint_vel_2", "joint_vel_3", "joint_vel_4", "joint_vel_5",
    "joint_current_0", "joint_current_1", "joint_current_2", "joint_current_3", "joint_current_4", "joint_current_5",
    "tcp_x", "tcp_y", "tcp_z",
    "tcp_force_x", "tcp_force_y", "tcp_force_z",
]


def compute_kl_divergence(baseline_df: pd.DataFrame, cf_df: pd.DataFrame, n_rows: int) -> float:
    """Compute KL divergence between baseline and CF pre-fault segments.

    Uses a histogram-based approximation of KL(baseline || cf) averaged across
    signal columns. Only compares the first n_rows of each (the pre-fault segment).
    Returns float('inf') if comparison is not possible.
    """
    import numpy as np

    cols = [c for c in _KL_SIGNAL_COLS if c in baseline_df.columns and c in cf_df.columns]
    if not cols:
        return float("inf")

    bl = baseline_df.iloc[:n_rows]
    cf = cf_df.iloc[:n_rows]

    if len(bl) < 5 or len(cf) < 5:
        return float("inf")

    kl_sum = 0.0
    valid_cols = 0

    for col in cols:
        bl_vals = pd.to_numeric(bl[col], errors="coerce").dropna().values
        cf_vals = pd.to_numeric(cf[col], errors="coerce").dropna().values

        if len(bl_vals) < 5 or len(cf_vals) < 5:
            continue

        # Shared bin edges from combined range
        all_vals = np.concatenate([bl_vals, cf_vals])
        n_bins = min(30, max(5, len(bl_vals) // 5))
        edges = np.linspace(all_vals.min() - 1e-9, all_vals.max() + 1e-9, n_bins + 1)

        # Compute histograms as probability distributions
        p, _ = np.histogram(bl_vals, bins=edges, density=True)
        q, _ = np.histogram(cf_vals, bins=edges, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()

        # KL(P || Q)
        kl = np.sum(p * np.log(p / q))
        kl_sum += kl
        valid_cols += 1

    if valid_cols == 0:
        return float("inf")
    return kl_sum / valid_cols


def _to_float(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_episode_df(
    ep_df: pd.DataFrame,
    first_timestamp_us: int,
) -> List[Dict[str, Any]]:
    """Convert a single episode DataFrame to list of UR3e schema dicts."""
    rows: List[Dict[str, Any]] = []

    for _, row in ep_df.iterrows():
        result: Dict[str, Any] = {}

        # Timestamp → ms since episode start
        t = row.get("time")
        if pd.notna(t):
            t_us = int(pd.Timestamp(t).value // 1000)  # ns -> us
            result["timestamp_ms"] = (t_us - first_timestamp_us) // 1000
        else:
            result["timestamp_ms"] = None

        # Map columns
        for schema_col, src_col in COLUMN_MAPPING.items():
            if src_col is None or src_col not in row.index:
                result[schema_col] = None
            else:
                val = row[src_col]
                if schema_col in _INT_COLS:
                    result[schema_col] = _to_int(val)
                else:
                    result[schema_col] = _to_float(val)

        # Extra fields for downstream use
        result["fault_label"] = _to_int(row.get("fault")) or 0
        result["task_phase"] = str(row.get("task_phase")) if pd.notna(row.get("task_phase")) else None

        rows.append(result)

    return rows


def load_episode_metadata(ep_row: pd.Series) -> Dict[str, Any]:
    """Parse episode metadata from the episode table row."""
    meta_str = ep_row.get("episode_metadata")
    if isinstance(meta_str, str):
        try:
            meta = ast.literal_eval(meta_str)
        except (ValueError, SyntaxError):
            meta = {}
    elif isinstance(meta_str, dict):
        meta = meta_str
    else:
        meta = {}

    return {
        "episode_id": str(ep_row.get("id", "")),
        "robot_type": meta.get("robot_model", "ur3"),
        "data_source": "factorywave",
        "task": meta.get("task", "unknown"),
        "condition": meta.get("condition", "unknown"),
        "fault_id": meta.get("fault_id"),
        "weight_of_box": meta.get("weight_of_box"),
        "shape_of_box": meta.get("shape_of_box"),
        "gripper_model": meta.get("gripper_model"),
        "payload_mass_configured": meta.get("payload_mass_configured"),
        "payload_cog_configured": meta.get("payload_cog_configured"),
        "tcp_offset_configured": meta.get("tcp_offset_configured"),
        "tcp_orientation_configured": meta.get("tcp_orientation_configured"),
        "counterfactual": bool(ep_row.get("counterfactual", False)),
        "cf_fault_id": _to_int(ep_row.get("cf_fault_id")),
        "cf_injection_timestep": _to_int(ep_row.get("cf_injection_timestep")),
        "cf_injection_time_s": _to_float(ep_row.get("cf_injection_time_s")),
        "cf_baseline_episode_id": str(ep_row.get("cf_baseline_episode_id"))
        if pd.notna(ep_row.get("cf_baseline_episode_id")) else None,
    }


def _decimate_episode(ep_df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    """Decimate an episode DataFrame to the target Hz.

    Preserves the exact fault onset row: if the 0→non-zero transition in the
    fault column falls between kept samples, the nearest decimated row is
    replaced with the original onset row so the transition boundary is exact.
    """
    ep_df = ep_df.reset_index(drop=True)
    times = ep_df["time"].values.astype("int64")
    if len(times) <= 5:
        return ep_df

    diffs = pd.Series(times).diff().dropna()
    pos_diffs = diffs[diffs > 0]
    if len(pos_diffs) <= 3:
        return ep_df

    med_dt_us = pos_diffs.median()
    hz = 1e6 / med_dt_us
    q = max(1, int(round(hz / target_hz)))
    if q <= 1:
        return ep_df

    # Find fault onset BEFORE decimation
    onset_idx = _find_fault_onset(ep_df)

    # Decimate
    continuous = set(ep_df.columns) - _NON_CONTINUOUS
    decimated = decimate_dataframe(ep_df, q=q, continuous_cols=continuous)

    # Splice in the original onset row if it wasn't kept
    if onset_idx is not None:
        onset_row = ep_df.iloc[[onset_idx]]
        # Find where it should go in the decimated frame
        decimated_onset = onset_idx // q
        if decimated_onset >= len(decimated):
            decimated_onset = len(decimated) - 1

        # Check if the decimated row at that position already has the transition
        dec_faults = pd.to_numeric(decimated["fault"], errors="coerce").fillna(0).values
        already_has_transition = (
            decimated_onset > 0
            and dec_faults[decimated_onset] != 0
            and dec_faults[decimated_onset - 1] == 0
        )

        if not already_has_transition and decimated_onset < len(decimated):
            # Replace the nearest row with the original onset row
            decimated = pd.concat([
                decimated.iloc[:decimated_onset],
                onset_row.reset_index(drop=True),
                decimated.iloc[decimated_onset + 1:],
            ], ignore_index=True)

    return decimated


def _find_fault_onset(ep_df: pd.DataFrame) -> Optional[int]:
    """Find the row index where the fault column becomes non-zero.

    Returns None if fault is never non-zero or is constant throughout.
    """
    faults = pd.to_numeric(ep_df["fault"], errors="coerce").fillna(0).values
    if faults[0] != 0:
        # Fault present from the start — check if it was absent at any point
        # (this means no pre-fault segment exists)
        return None
    for i in range(1, len(faults)):
        if faults[i] != 0:
            return i
    return None


def normalize_dataset(
    input_dir: Path,
    output_dir: Path,
    target_hz: int = TARGET_HZ,
    limit: Optional[int] = None,
    tasks: Optional[List[str]] = None,
) -> None:
    """Normalize FactoryWave ur_signals to per-episode UR3e schema JSON files.

    For counterfactual episodes: selects the best CF per baseline group using
    KL divergence on the pre-fault segment, then outputs as
    {"baseline": [...], "counterfactual": [...]}.

    For other episodes: outputs as a flat list of rows.
    """
    import numpy as np

    # Load episode table
    episode_path = input_dir / "episode.parquet"
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode table not found: {episode_path}")

    logger.info("Loading episode metadata...")
    ep_table = pd.read_parquet(episode_path)

    # Parse metadata to filter by task
    ep_table["_meta"] = ep_table["episode_metadata"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else (x or {})
    )
    ep_table["_task"] = ep_table["_meta"].apply(lambda m: m.get("task") if isinstance(m, dict) else None)
    ep_table["_condition"] = ep_table["_meta"].apply(lambda m: m.get("condition") if isinstance(m, dict) else None)

    if tasks:
        ep_table = ep_table[ep_table["_task"].isin(tasks)]
        logger.info(f"Filtered to tasks {tasks}: {len(ep_table)} episodes")

    ep_lookup = {str(row["id"]): row for _, row in ep_table.iterrows()}

    # Build counterfactual groups: baseline_id -> [cf_episode_ids]
    cf_groups: Dict[str, List[str]] = {}
    cf_episode_ids: set = set()
    for _, row in ep_table.iterrows():
        if row["_condition"] == "counterfactual" and pd.notna(row.get("cf_baseline_episode_id")):
            bl_id = str(row["cf_baseline_episode_id"])
            cf_id = str(row["id"])
            cf_groups.setdefault(bl_id, []).append(cf_id)
            cf_episode_ids.add(cf_id)

    baseline_ids = set(cf_groups.keys())
    logger.info(f"Episode metadata loaded: {len(ep_lookup)} episodes, "
                f"{len(cf_groups)} CF groups ({len(cf_episode_ids)} CF episodes)")

    # Find signal parquet files
    signal_paths = []
    for subdir in ["ur_signals_10hz", "ur_signals_125hz"]:
        p = input_dir / subdir / "data.parquet"
        if p.exists():
            signal_paths.append(p)
    if not signal_paths:
        single = input_dir / "ur_signals.parquet"
        if single.exists():
            signal_paths.append(single)
    if not signal_paths:
        raise FileNotFoundError(f"No ur_signals parquet found in {input_dir}")

    logger.info(f"Signal files: {[str(p) for p in signal_paths]}")

    out_dir = output_dir / "factorywave"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Pass 1: Load and decimate all episodes, store in memory by ID
    # ---------------------------------------------------------------
    logger.info("Pass 1: Loading and decimating episodes...")
    episode_dfs: Dict[str, pd.DataFrame] = {}

    for signal_path in signal_paths:
        pf = pq.ParquetFile(signal_path)
        for rg_idx in range(pf.metadata.num_row_groups):
            df = pf.read_row_group(rg_idx).to_pandas(types_mapper=lambda t: None)
            for ep_id, ep_df in df.groupby("episode_id", sort=False):
                ep_id_str = str(ep_id)
                if ep_id_str not in ep_lookup:
                    continue
                # Accumulate rows for episodes that span row groups
                if ep_id_str in episode_dfs:
                    episode_dfs[ep_id_str] = pd.concat(
                        [episode_dfs[ep_id_str], ep_df], ignore_index=True
                    )
                else:
                    episode_dfs[ep_id_str] = ep_df.reset_index(drop=True)

    logger.info(f"  Loaded {len(episode_dfs)} episodes into memory")

    # Decimate all episodes to target Hz
    for ep_id_str, ep_df in episode_dfs.items():
        episode_dfs[ep_id_str] = _decimate_episode(ep_df, target_hz)

    logger.info(f"  Decimated to {target_hz} Hz")

    # ---------------------------------------------------------------
    # Pass 2: Process counterfactual groups (KL selection)
    # ---------------------------------------------------------------
    logger.info("Pass 2: Selecting best counterfactual per baseline (KL divergence)...")
    cf_processed = 0
    cf_skipped = 0

    for baseline_id, cf_ids in cf_groups.items():
        if baseline_id not in episode_dfs:
            cf_skipped += len(cf_ids)
            continue

        out_file = out_dir / f"{baseline_id}.json"
        if out_file.exists():
            cf_processed += 1
            continue

        baseline_df = episode_dfs[baseline_id]

        # Evaluate each CF: use cf_injection_timestep as the fault onset
        candidates: List[Tuple[str, float, int]] = []  # (cf_id, kl_score, onset_idx)
        for cf_id in cf_ids:
            if cf_id not in episode_dfs:
                continue
            cf_df = episode_dfs[cf_id]
            cf_row = ep_lookup.get(cf_id)
            if cf_row is None:
                continue

            # Detect fault onset from the signal's fault column (0 → non-zero).
            # This is the most reliable method after decimation.
            onset = _find_fault_onset(cf_df)

            # Fallback to metadata if signal doesn't have a transition
            if onset is None:
                raw_timestep = cf_row.get("cf_injection_timestep")
                if pd.notna(raw_timestep) and raw_timestep is not None:
                    onset = max(1, int(float(raw_timestep) / (500.0 / target_hz)))

            if onset is None or onset < 5 or onset >= len(cf_df):
                continue

            # Compute KL on pre-fault segment
            n_rows = min(onset, len(baseline_df))
            kl = compute_kl_divergence(baseline_df, cf_df, n_rows)
            candidates.append((cf_id, kl, onset))

        if not candidates:
            cf_skipped += 1
            continue

        # Select the CF with minimum KL divergence
        best_cf_id, best_kl, best_onset = min(candidates, key=lambda x: x[1])
        best_cf_df = episode_dfs[best_cf_id]

        # Normalize both
        bl_first_t = int(pd.Timestamp(baseline_df["time"].values[0]).value // 1000)
        cf_first_t = int(pd.Timestamp(best_cf_df["time"].values[0]).value // 1000)

        baseline_rows = normalize_episode_df(baseline_df, bl_first_t)
        cf_rows = normalize_episode_df(best_cf_df, cf_first_t)

        if not baseline_rows or not cf_rows:
            cf_skipped += 1
            continue

        # Write as combined baseline + counterfactual
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"baseline": baseline_rows, "counterfactual": cf_rows}, f, indent=2, cls=_NaNSafeEncoder)

        # Metadata
        bl_row = ep_lookup[baseline_id]
        cf_row = ep_lookup[best_cf_id]
        metadata = load_episode_metadata(bl_row)
        metadata["baseline"] = {
            "num_samples": len(baseline_rows),
            "last_timestamp_ms": baseline_rows[-1]["timestamp_ms"],
        }
        metadata["counterfactual"] = {
            "episode_id": best_cf_id,
            "num_samples": len(cf_rows),
            "last_timestamp_ms": cf_rows[-1]["timestamp_ms"],
            "fault_onset_index": best_onset,
            "kl_divergence": round(best_kl, 6),
            "cf_fault_id": _to_int(cf_row.get("cf_fault_id")),
            "cf_injection_timestep": _to_int(cf_row.get("cf_injection_timestep")),
            "candidates_evaluated": len(candidates),
        }
        metadata["schema_version"] = "1.0"

        meta_file = out_dir / f"{baseline_id}_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        cf_processed += 1
        if cf_processed % 50 == 0:
            logger.info(f"  {cf_processed} CF groups processed...")

    logger.info(f"  CF groups: {cf_processed} processed, {cf_skipped} skipped")

    # ---------------------------------------------------------------
    # Pass 3: Process non-counterfactual episodes (normal, fault, trajectory-opt)
    # ---------------------------------------------------------------
    logger.info("Pass 3: Normalizing non-counterfactual episodes...")
    regular_processed = 0
    regular_skipped = 0

    for ep_id_str, ep_df in episode_dfs.items():
        # Skip CFs and baselines (already handled)
        if ep_id_str in cf_episode_ids or ep_id_str in baseline_ids:
            continue

        if ep_id_str not in ep_lookup:
            continue

        if limit and regular_processed >= limit:
            break

        out_file = out_dir / f"{ep_id_str}.json"
        if out_file.exists():
            regular_processed += 1
            continue

        # Normalize
        first_t = int(pd.Timestamp(ep_df["time"].values[0]).value // 1000)
        normalized_rows = normalize_episode_df(ep_df, first_t)

        if not normalized_rows:
            regular_skipped += 1
            continue

        # Write as flat list
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(normalized_rows, f, indent=2, cls=_NaNSafeEncoder)

        # Metadata
        ep_row = ep_lookup[ep_id_str]
        metadata = load_episode_metadata(ep_row)
        metadata["num_samples"] = len(normalized_rows)
        metadata["schema_version"] = "1.0"
        metadata["timestamp_info"] = {
            "format": "milliseconds since episode start",
            "last_timestamp_ms": normalized_rows[-1]["timestamp_ms"],
        }

        meta_file = out_dir / f"{ep_id_str}_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        regular_processed += 1
        if regular_processed % 100 == 0:
            logger.info(f"  {regular_processed} episodes normalized...")

    logger.info(f"  Regular episodes: {regular_processed} processed, {regular_skipped} skipped")
    logger.info(f"Done. Output: {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize FactoryWave dataset to UR3e schema JSON."
    )
    repo_root = Path(__file__).resolve().parents[2]

    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "data" / "factorywave" / "data",
        help="Input directory containing episode.parquet and ur_signals parquet files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "normalized_episodes",
        help="Output directory for normalized JSON files",
    )
    parser.add_argument(
        "--target-hz",
        type=int,
        default=TARGET_HZ,
        help=f"Target sampling rate after decimation (default: {TARGET_HZ})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of episodes to normalize (for testing)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Filter by task (e.g. pick_and_place peg_in_hole)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        normalize_dataset(
            input_dir=args.input,
            output_dir=args.output,
            target_hz=args.target_hz,
            limit=args.limit,
            tasks=args.tasks,
        )
        return 0
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
