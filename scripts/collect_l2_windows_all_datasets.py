#!/usr/bin/env python3
"""Collect L2 event windows from every dataset, not just the normalized ones.

The threshold calibration for L2 template 3 has to see the whole corpus. Only
factorywave/KUKA is present under ``data/normalized_episodes``, so constants
fitted from that directory alone are KUKA medians and will not sit at 50% on
UR3e, the UR screwdriver cell, or the two open datasets. Those robots record
different channels (KUKA has no velocity, contact force or TCP setpoint at
all), and where a channel is shared its scale differs.

This module normalizes a sample of episodes straight from the raw sources,
using the same column mappings the real normalizers use, and yields windows in
the exact shape ``sample_subseries_before_event`` expects. It is read-only: no
normalized episode is written to disk.

Sources:
  factorywave_kuka   data/normalized_episodes/factorywave  (already normalized)
  factorywave_ur     data/factorywave/data/ur_signals.parquet
  aursad             data/open_datasets/aursad/experiment_*.parquet
  vorausad           data/open_datasets/vorausad/*.parquet

Usage:
    python scripts/collect_l2_windows_all_datasets.py --per-dataset 120
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.question_generation.level2.level2 import (  # noqa: E402
    CONTEXT_MAX,
    CONTEXT_MIN,
    MIN_POST_EVENT_TIMESTAMPS_AFTER,
    _first_timestamp_ms,
    normalize_timestamps,
    sample_subseries_before_event,
    split_event_segment,
)

REPO = Path(__file__).resolve().parents[1]


def _load_mapping(name: str) -> Dict[str, Optional[str]]:
    path = REPO / "data" / "mappings_of_features" / f"{name}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("mapping", raw)


def _factorywave_mapping() -> Dict[str, Optional[str]]:
    from src.data.factorywave_normalizer import _build_column_mapping
    return _build_column_mapping()


def _rows_from_frame(df, mapping: Dict[str, Optional[str]], fault_id: Any) -> List[Dict[str, Any]]:
    """Turn a raw episode frame into normalized row dicts.

    Only the canonical fields the mc statements read are emitted. ``event`` is
    synthesised from the episode's fault: the generator's window sampler keys
    on an ``event`` column going 0 -> non-zero, and the raw frames carry the
    fault as episode-level metadata rather than a per-row flag.
    """
    cols = set(df.columns)
    usable = {canon: src for canon, src in mapping.items() if src and src in cols}
    if not usable:
        return []
    n = len(df)
    if n < CONTEXT_MIN + MIN_POST_EVENT_TIMESTAMPS_AFTER + 2:
        return []

    series = {canon: df[src].tolist() for canon, src in usable.items()}
    onset = n // 2  # fault flips on at mid-episode, giving pre and post context
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        row: Dict[str, Any] = {"timestamp_ms": i * 100}
        for canon, values in series.items():
            v = values[i]
            # pandas nullable dtypes yield pd.NA, whose truthiness raises, so
            # coerce through float() and treat any failure as missing.
            try:
                fv = float(v)
                row[canon] = None if fv != fv else fv
            except (TypeError, ValueError):
                row[canon] = None
        row["event"] = 0 if i < onset else int(fault_id or 1)
        rows.append(row)
    return rows


def _iter_parquet_episodes(paths: List[Path], mapping, limit: int, rng) -> Iterator[Tuple[list, dict]]:
    import pyarrow.parquet as pq
    rng.shuffle(paths)
    yielded = 0
    for path in paths:
        if yielded >= limit:
            return
        try:
            table = pq.ParquetFile(path).read()
            df = table.to_pandas()
        except Exception:
            continue
        rows = _rows_from_frame(df, mapping, fault_id=1)
        if rows:
            yield rows, {"fault_id": 1, "source": path.stem}
            yielded += 1


def _iter_grouped_parquet(path: Path, mapping, limit: int, rng, group_col="episode_id"):
    """factorywave UR ships one big table keyed by episode_id."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    if group_col not in pf.schema.names:
        return
    needed = [c for c in {*(v for v in mapping.values() if v), group_col} if c in pf.schema.names]
    seen: Dict[Any, List[int]] = {}
    frames = []
    for batch in pf.iter_batches(batch_size=200_000, columns=needed):
        frames.append(batch.to_pandas())
        if sum(len(f) for f in frames) > 2_000_000:
            break
    if not frames:
        return
    import pandas as pd
    df = pd.concat(frames, ignore_index=True)
    groups = list(df.groupby(group_col))
    rng.shuffle(groups)
    for gid, gdf in groups[: limit * 3]:
        rows = _rows_from_frame(gdf, mapping, fault_id=1)
        if rows:
            yield rows, {"fault_id": 1, "source": str(gid)}
            limit -= 1
            if limit <= 0:
                return


def _windows_from_rows(rows: List[Dict[str, Any]], meta: dict):
    try:
        sampled = sample_subseries_before_event(
            rows, CONTEXT_MIN, CONTEXT_MAX,
            min_post_event_after=MIN_POST_EVENT_TIMESTAMPS_AFTER,
            return_metadata=True,
        )
    except Exception:
        return None
    subseries, post_event_rows = sampled[0], sampled[1]
    if not subseries or not post_event_rows:
        return None
    base = _first_timestamp_ms(subseries)
    subseries = normalize_timestamps(subseries, base)
    post_event_rows = normalize_timestamps(post_event_rows, base)
    event_rows, _ = split_event_segment(post_event_rows)
    if not event_rows:
        return None
    return subseries, post_event_rows, meta


def collect_all(per_dataset: int = 120, seed: int = 42) -> List[Tuple[list, list, dict]]:
    """Event windows pooled across every dataset, tagged with `dataset` in meta."""
    rng = random.Random(seed)
    out: List[Tuple[list, list, dict]] = []

    # 1. factorywave KUKA, already normalized on disk
    from scripts.calibrate_l2_tmpl3_thresholds import collect_windows as kuka_windows
    for sub, post, meta in kuka_windows(REPO / "data" / "normalized_episodes", per_dataset, seed):
        meta = dict(meta); meta["dataset"] = "factorywave_kuka"
        out.append((sub, post, meta))

    # 2. factorywave UR, from the raw signal table
    ur = REPO / "data" / "factorywave" / "data" / "ur_signals.parquet"
    if ur.is_file():
        for rows, meta in _iter_grouped_parquet(ur, _factorywave_mapping(), per_dataset, rng):
            w = _windows_from_rows(rows, {**meta, "dataset": "factorywave_ur"})
            if w:
                out.append(w)

    # 3 + 4. the two open datasets
    for name, folder in [("aursad", "aursad"), ("vorausad", "vorausad")]:
        d = REPO / "data" / "open_datasets" / folder
        if not d.is_dir():
            continue
        paths = sorted(d.glob("*.parquet"))
        if not paths:
            continue
        for rows, meta in _iter_parquet_episodes(paths, _load_mapping(name), per_dataset, rng):
            w = _windows_from_rows(rows, {**meta, "dataset": name})
            if w:
                out.append(w)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-dataset", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    w = collect_all(args.per_dataset, args.seed)
    print(f"collected {len(w)} windows")
    for ds, n in sorted(Counter(m.get("dataset") for _, _, m in w).items()):
        print(f"  {ds:20s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
