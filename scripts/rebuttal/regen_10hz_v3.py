"""Regenerate every impossible-at-10Hz Q&A item using the existing
per-source normalizers (which own the correct column mappings).

This version calls into `src/data/factorywave_normalizer.py` (UR track),
`src/data/factorywave_kuka_normalizer.py` (KUKA track), and reads the
JSON mappings under `data/mappings_of_features/` for AURSAD and voraus.
The output for every source is a list of dicts on the FactoryBench UR3e
schema (feedback_pos_N, setpoint_pos_N, est_contact_force_N, ...).

Usage:
    python scripts/rebuttal/regen_10hz_v3.py --smoke
    python scripts/rebuttal/regen_10hz_v3.py --full
    python scripts/rebuttal/regen_10hz_v3.py --full --push
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.data.factorywave_normalizer import (
    normalize_episode_df, _decimate_episode, TARGET_HZ,
)
from src.data.factorywave_kuka_normalizer import (
    normalize_kuka_episode_df, _decimate_kuka_episode,
)

DATA_DIR = Path("data")

FW_PARQUETS = {
    "ur":             DATA_DIR / "factorywave/data/ur_signals.parquet",
    "ur_screwdriver": DATA_DIR / "factorywave/data/ur_screwdriver_signals.parquet",
    "kuka":           DATA_DIR / "factorywave/data/kuka_signals.parquet",
}
FW_EPISODES_META = DATA_DIR / "factorywave/data/episode.parquet"
AURSAD_DIR = DATA_DIR / "open_datasets/aursad"
VORAUS_DIR = DATA_DIR / "open_datasets/vorausad"
AURSAD_MAP = json.loads((DATA_DIR / "mappings_of_features/aursad.json").read_text())["mapping"]
VORAUS_MAP = json.loads((DATA_DIR / "mappings_of_features/vorausad.json").read_text())["mapping"]

TARGET_DT_MS = int(round(1000.0 / TARGET_HZ))   # 100


# ---------------- parquet caching (episode-index only; body read on demand via pyarrow filter) ----------------
_FW_EP_INDEX: Dict[str, set] = {}
_FW_META_DF: Optional[pd.DataFrame] = None


def _load_fw_index(track: str) -> set:
    if track in _FW_EP_INDEX: return _FW_EP_INDEX[track]
    import pyarrow.parquet as pq
    tbl = pq.read_table(FW_PARQUETS[track], columns=["episode_id"])
    _FW_EP_INDEX[track] = set(tbl.column("episode_id").to_pylist())
    print(f"  indexed {FW_PARQUETS[track].name}: {len(_FW_EP_INDEX[track])} episodes", flush=True)
    return _FW_EP_INDEX[track]


def _load_fw_episode_df(track: str, episode_id: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    tbl = pq.read_table(FW_PARQUETS[track], filters=[("episode_id", "=", episode_id)])
    return tbl.to_pandas().sort_values("time").reset_index(drop=True)


def _load_fw_meta() -> pd.DataFrame:
    global _FW_META_DF
    if _FW_META_DF is None and FW_EPISODES_META.exists():
        _FW_META_DF = pd.read_parquet(FW_EPISODES_META)
    return _FW_META_DF if _FW_META_DF is not None else pd.DataFrame()


def _fw_metadata_fault_id(episode_id: str) -> Optional[int]:
    meta = _load_fw_meta()
    if meta.empty or "id" not in meta.columns:
        return None
    hits = meta[meta["id"] == episode_id]
    if hits.empty:
        return None
    row = hits.iloc[0]
    for k in ("fault_id", "fault", "primary_fault_id"):
        v = row.get(k)
        if v is not None and pd.notna(v):
            try: return int(v)
            except Exception: pass
    return None


# ---------------- episode loaders (return schema-named rows at 10 Hz) ----------------
def _fw_episode_rows(episode_id: str) -> Optional[List[Dict[str, Any]]]:
    for track in ("ur", "ur_screwdriver", "kuka"):
        if episode_id not in _load_fw_index(track):
            continue
        ep_df = _load_fw_episode_df(track, episode_id)
        if ep_df.empty:
            return None
        fault_id = _fw_metadata_fault_id(episode_id)
        if track == "kuka":
            dec = _decimate_kuka_episode(ep_df, TARGET_HZ)
            first_ts_us = int(pd.Timestamp(dec["time"].iloc[0]).value // 1000)
            return normalize_kuka_episode_df(dec, first_ts_us, fault_id)
        else:
            dec = _decimate_episode(ep_df, TARGET_HZ)
            first_ts_us = int(pd.Timestamp(dec["time"].iloc[0]).value // 1000)
            return normalize_episode_df(dec, first_ts_us, fault_id)
    return None


def _snap_to_10hz_via_stride(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Snap already-loaded schema rows to a strict 10 ms lattice."""
    if len(rows) < 2: return rows
    ts = np.array([r["timestamp_ms"] for r in rows], dtype=np.int64)
    dt_med = float(np.median(np.diff(ts))) if len(ts) > 1 else 100.0
    if 90 <= dt_med <= 110:
        for i, r in enumerate(rows):
            r["timestamp_ms"] = i * TARGET_DT_MS
        return rows
    stride = max(1, int(round(TARGET_DT_MS / dt_med)))
    out = rows[::stride]
    for i, r in enumerate(out):
        r["timestamp_ms"] = i * TARGET_DT_MS
    return out


def _generic_episode_rows(df: pd.DataFrame, mapping: Dict[str, Any],
                          ts_col: str, native_hz: float) -> List[Dict[str, Any]]:
    """For AURSAD / voraus: apply the schema mapping and snap to 10Hz. Vectorised."""
    n = len(df)
    if ts_col in df.columns:
        col = df[ts_col]
        if pd.api.types.is_datetime64_any_dtype(col):
            if getattr(col.dt, "tz", None) is not None:
                col = col.dt.tz_convert("UTC").dt.tz_localize(None)
            ts_ns = col.astype("datetime64[ns]").astype(np.int64).to_numpy()
            ts_ms = (ts_ns - ts_ns[0]) // 1_000_000
        else:
            arr = np.asarray(col.to_list(), dtype=np.int64)
            ts_ms = arr - arr[0]
    else:
        ts_ms = (np.arange(n) * (1000.0 / max(native_hz, 1))).astype(np.int64)

    # Precompute per-schema-column numpy arrays once, avoiding per-cell df.iloc lookups
    schema_arrays: Dict[str, Optional[np.ndarray]] = {}
    for schema_col, src in mapping.items():
        if schema_col == "timestamp_ms": continue
        if src is None or src not in df.columns:
            schema_arrays[schema_col] = None
        else:
            schema_arrays[schema_col] = df[src].to_numpy()

    schema_rows: List[Dict[str, Any]] = []
    for i in range(n):
        r: Dict[str, Any] = {"timestamp_ms": int(ts_ms[i])}
        for schema_col, arr in schema_arrays.items():
            if arr is None:
                r[schema_col] = None
                continue
            v = arr[i]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                r[schema_col] = None
            elif isinstance(v, (int, float, np.integer, np.floating)):
                r[schema_col] = float(v)
            else:
                r[schema_col] = v
        schema_rows.append(r)
    return _snap_to_10hz_via_stride(schema_rows)


def _aursad_episode_rows(episode_id: str) -> Optional[List[Dict[str, Any]]]:
    p = AURSAD_DIR / f"{episode_id}.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p)
    return _generic_episode_rows(df, AURSAD_MAP, ts_col="timestamp", native_hz=10.0)


def _voraus_episode_rows(episode_id: str) -> Optional[List[Dict[str, Any]]]:
    p = VORAUS_DIR / f"{episode_id}.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p)
    return _generic_episode_rows(df, VORAUS_MAP, ts_col="timestamp_ms", native_hz=50.0)


def load_episode(dataset: str, episode_id: str) -> Optional[List[Dict[str, Any]]]:
    ds = (dataset or "").lower()
    if ds.startswith("factorywave"):   # covers both "factorywave" and "factorywave_kuka"
        return _fw_episode_rows(episode_id)
    if ds == "aursad":
        return _aursad_episode_rows(episode_id)
    if "voraus" in ds:
        return _voraus_episode_rows(episode_id)
    return None


# ---------------- item regeneration ----------------
_ACRONYMS = {
    "effort_target_torque_": "ett", "feedback_pos_": "fp", "setpoint_pos_": "sp",
    "feedback_speed_": "fs", "est_contact_force_": "ecf", "timestamp_ms": "tm",
    "effort_current_": "ec", "effort_target_current_": "etc",
    "control_output_": "co", "joint_temp_": "jt",
    "feedback_tcp_": "ftcp", "setpoint_tcp_": "stcp",
    "true_force_": "tf", "vibration_": "vib",
}


def _short(name: str) -> str:
    for prefix, short in _ACRONYMS.items():
        if name.startswith(prefix):
            return short + name[len(prefix):]
    return name


def _rebuild_context(rows_10hz: List[Dict[str, Any]], start: int, length: int,
                     visible_signals: Optional[List[str]] = None) -> Dict[str, Any]:
    sub = rows_10hz[start:start + length]
    if not sub: return {"time_series": []}
    if visible_signals:
        visible_set = set(visible_signals) | {"timestamp_ms"}
    else:
        visible_set = None
    lines = []
    for r in sub:
        ts = int(r.get("timestamp_ms", 0))
        parts = []
        for k, v in r.items():
            if k == "timestamp_ms": continue
            if visible_set is not None and k not in visible_set: continue
            if v is None: continue
            if isinstance(v, (int, float)):
                parts.append(f"{_short(k)}={round(v, 4)}")
        lines.append(f"t={ts}: " + ", ".join(parts))
    return {"time_series": lines}


def _visible_signals_from_context(item: Dict[str, Any]) -> Optional[List[str]]:
    ts_rows = (item.get("context") or {}).get("time_series") or []
    if not ts_rows: return None
    first = ts_rows[0] if isinstance(ts_rows[0], str) else ""
    # extract "k=v" tokens after "t=<ts>:"
    keys = re.findall(r"([A-Za-z][A-Za-z0-9_]*)=", first.split(":", 1)[-1])
    _INV = {short: prefix for prefix, short in _ACRONYMS.items()}
    out = set()
    for k in keys:
        # try to expand short -> long
        matched = None
        for short in sorted(_INV, key=len, reverse=True):
            if k == short: matched = _INV[short].rstrip("_"); break
            if k.startswith(short):
                matched = _INV[short] + k[len(short):]; break
        out.add(matched if matched else k)
    return sorted(out) if out else None


def _rewrite_question(q: str, n_ms: int) -> str:
    return re.sub(r"T\+\d+\s*ms", f"T+{n_ms}ms", q, count=1)


def regen_l1_t7(item: Dict[str, Any], rows_10hz: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    prov = item.get("provenance") or {}
    start = int(prov.get("subseries_start_index", 0))
    length = int(prov.get("subseries_length", 0))
    bounds = dict(item.get("acceptance_bounds") or {})
    signal = bounds.get("signal")
    steps = int(bounds.get("steps_ahead", 1))
    if not signal or start < 0 or length <= 0: return None
    future_idx = start + length + steps - 1
    if future_idx >= len(rows_10hz) or start + length > len(rows_10hz): return None
    future_row = rows_10hz[future_idx]
    last_ctx = rows_10hz[start + length - 1]
    if signal not in future_row or future_row[signal] is None: return None
    n_ms = int(future_row["timestamp_ms"]) - int(last_ctx["timestamp_ms"])
    new_answer = round(float(future_row[signal]), 4)
    out = dict(item)
    out["question"] = _rewrite_question(item.get("question", ""), n_ms)
    out["answer"] = new_answer
    out["acceptance_bounds"] = {**bounds, "actual_value": new_answer, "horizon_ms": n_ms}
    out["context"] = _rebuild_context(rows_10hz, start, length, _visible_signals_from_context(item))
    return out


def _infer_horizon_ms(item: Dict[str, Any], bounds: Dict[str, Any]) -> Optional[int]:
    if bounds.get("horizon_ms"): return int(bounds["horizon_ms"])
    m = re.search(r"T\+(\d+)\s*ms", item.get("question", ""))
    return int(m.group(1)) if m else None


def regen_l2_predict(item: Dict[str, Any], rows_10hz: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    prov = item.get("provenance") or {}
    start = int(prov.get("subseries_start_index", 0))
    length = int(prov.get("subseries_length", 0))
    bounds = dict(item.get("acceptance_bounds") or {})
    horizon_ms = _infer_horizon_ms(item, bounds)
    if horizon_ms is None or start < 0 or length <= 0: return None
    steps_ahead = max(1, min(10, int(round(horizon_ms / TARGET_DT_MS))))
    future_idx = start + length + steps_ahead - 1
    if future_idx >= len(rows_10hz): return None
    future_row = rows_10hz[future_idx]
    last_ctx = rows_10hz[start + length - 1]
    n_ms = int(future_row["timestamp_ms"]) - int(last_ctx["timestamp_ms"])
    signal = bounds.get("signal")
    is_tensor = isinstance(bounds.get("std"), list)
    if is_tensor:
        base = signal
        if not isinstance(base, str): return None
        vals, stds = [], []
        for j in range(6):
            key = f"{base}_{j}"
            v = future_row.get(key)
            if v is None: return None
            vals.append(round(float(v), 6))
            seg = [r.get(key) for r in rows_10hz[start:start + length] if r.get(key) is not None]
            stds.append(round(float(np.std(seg)) if seg else 0.0, 6))
        new_answer = "[" + ",".join(str(v) for v in vals) + "]"
        new_bounds = {**bounds, "std": stds,
                      "margin": [round(s * 0.75, 6) for s in stds],
                      "horizon_ms": n_ms}
    else:
        if not signal: return None
        v = future_row.get(signal)
        if v is None: return None
        new_answer = round(float(v), 6)
        seg = [r.get(signal) for r in rows_10hz[start:start + length] if r.get(signal) is not None]
        std = float(np.std(seg)) if seg else 0.0
        new_bounds = {**bounds, "std": round(std, 6),
                      "margin": round(std * 0.75, 6),
                      "horizon_ms": n_ms}
    out = dict(item)
    out["question"] = _rewrite_question(item.get("question", ""), n_ms)
    out["answer"] = new_answer
    out["acceptance_bounds"] = new_bounds
    out["context"] = _rebuild_context(rows_10hz, start, length, _visible_signals_from_context(item))
    return out


def regen_l3(item: Dict[str, Any], rows_10hz: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return regen_l2_predict(item, rows_10hz)


# ---------------- HF wrapper ----------------
class _HF:
    def __init__(self):
        load_dotenv(find_dotenv(usecwd=True))
        from huggingface_hub import hf_hub_download, HfApi
        self._hf = hf_hub_download
        self._api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
        self.tok = os.environ["HF_WRITE_TOKEN"]
    def dl(self, filename: str) -> str:
        return self._hf("FactoryBench/FactoryBench", filename, repo_type="dataset", token=self.tok)
    def push_file(self, local: str, remote: str, msg: str) -> None:
        self._api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                              repo_id="FactoryBench/FactoryBench", repo_type="dataset",
                              commit_message=msg)


# ---------------- driver ----------------
def is_impossible(item: Dict[str, Any]) -> bool:
    m = re.search(r"T\+(\d+)\s*ms", item.get("question", ""))
    if not m: return False
    N = int(m.group(1))
    level = item.get("level")
    if level == 1: return True
    if N % 100 != 0: return True
    return False


def _debug_fail(item, rows, level):
    prov = item.get("provenance") or {}
    bounds = item.get("acceptance_bounds") or {}
    start = prov.get("subseries_start_index"); length = prov.get("subseries_length")
    signal = bounds.get("signal"); steps = bounds.get("steps_ahead")
    fut = None
    try: fut = int(start) + int(length) + int(steps or 1) - 1
    except Exception: pass
    sig_in_row0 = "?"
    if rows and signal:
        if isinstance(signal, str):
            sig_in_row0 = signal in rows[0] if not isinstance(bounds.get("std"), list) else f"{signal}_0" in rows[0]
    print(f"    [dbg L{level}] id={(item.get('id') or '')[:8]} ds={prov.get('dataset')} "
          f"ep={(prov.get('episode') or prov.get('sampled_subfolder','') or '')[:20]} "
          f"start={start} len={length} steps={steps} future_idx={fut} "
          f"rows_10hz={len(rows) if rows else 0} sig={signal} sig_in_row0={sig_in_row0}")


def _bulk_prefetch(needed: Dict[str, set]) -> Dict[Tuple[str, str], Optional[List[Dict[str, Any]]]]:
    """Load every requested episode in bulk, one parquet-scan per FactoryWave track,
    and a thread pool for per-file (AURSAD/voraus) sources. Returns a cache keyed
    by (dataset, episode_id) with resampled+schema-normalized rows.
    """
    cache: Dict[Tuple[str, str], Optional[List[Dict[str, Any]]]] = {}

    # --- FactoryWave bulk-scan per track ---
    fw_eps = set()
    for ds_lower in list(needed):
        if ds_lower.startswith("factorywave"):
            fw_eps |= needed[ds_lower]
    if fw_eps:
        import pyarrow.parquet as pq
        for track, filename in FW_PARQUETS.items():
            here = fw_eps & _load_fw_index(track)
            if not here:
                continue
            print(f"  bulk-loading {len(here)} episodes from {filename.name} ...", flush=True)
            here_list = list(here)
            # PyArrow "in" filter: use isin via chunked filter
            tbl = pq.read_table(filename, filters=[("episode_id", "in", here_list)])
            df_all = tbl.to_pandas()
            # group and normalize each episode
            for ep_id, ep_df in df_all.groupby("episode_id", sort=False):
                ep_df = ep_df.sort_values("time").reset_index(drop=True)
                if ep_df.empty:
                    continue
                fault_id = _fw_metadata_fault_id(ep_id)
                try:
                    if track == "kuka":
                        dec = _decimate_kuka_episode(ep_df, TARGET_HZ)
                        first_ts_us = int(pd.Timestamp(dec["time"].iloc[0]).value // 1000)
                        rows = normalize_kuka_episode_df(dec, first_ts_us, fault_id)
                    else:
                        dec = _decimate_episode(ep_df, TARGET_HZ)
                        first_ts_us = int(pd.Timestamp(dec["time"].iloc[0]).value // 1000)
                        rows = normalize_episode_df(dec, first_ts_us, fault_id)
                except Exception as e:
                    print(f"    [bulk fail] {track}/{ep_id[:12]}: {type(e).__name__}: {e}", flush=True)
                    rows = None
                # both possible dataset spellings alias to the same rows
                cache[("factorywave", ep_id)] = rows
                cache[("factorywave_kuka", ep_id)] = rows
                fw_eps.discard(ep_id)
            del df_all, tbl
            import gc; gc.collect()

    # --- Per-file sources (AURSAD, voraus): thread pool ---
    from concurrent.futures import ThreadPoolExecutor
    per_file_jobs: List[Tuple[str, str]] = []
    for ds_lower, eps in needed.items():
        if ds_lower == "aursad" or "voraus" in ds_lower:
            for ep in eps:
                per_file_jobs.append((ds_lower, ep))
    if per_file_jobs:
        print(f"  loading {len(per_file_jobs)} per-file episodes (aursad/voraus) with 8 threads ...", flush=True)
        def _load_one(job):
            ds_l, ep = job
            try:
                if ds_l == "aursad":  return job, _aursad_episode_rows(ep)
                else:                 return job, _voraus_episode_rows(ep)
            except Exception:
                return job, None
        with ThreadPoolExecutor(max_workers=8) as pool:
            for job, rows in pool.map(_load_one, per_file_jobs, chunksize=16):
                cache[job] = rows

    # fill in any misses so callers don't re-attempt
    for ds_lower, eps in needed.items():
        for ep in eps:
            cache.setdefault((ds_lower, ep), None)
    return cache


def regen_split(level: int, split: str, hf: _HF, limit: Optional[int] = None,
                incremental_path: Optional[Path] = None,
                progress_every: int = 100,
                flush_every: int = 100,
                ep_cache_max: int = 200) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Regenerate a whole split.

    If ``incremental_path`` is set, every item (kept + regenerated) is appended to
    that file as it is processed, so the on-disk state survives a mid-split kill.
    A ``.done`` sibling marker is written after the split finishes cleanly.
    Every ``progress_every`` candidates a `[progress]` line prints so wall-clock
    is observable in the log.
    """
    import gc
    from collections import OrderedDict
    stats = defaultdict(int)
    path = hf.dl(f"factorybench_qa/level_{level}/{split}.jsonl")
    src = [json.loads(ln) for ln in open(path, encoding="utf-8")]
    out: List[Dict[str, Any]] = []

    # ---- pre-scan: collect the set of episodes we actually need, then bulk-load ----
    needed: Dict[str, set] = defaultdict(set)
    for it in src:
        if not is_impossible(it): continue
        prov = it.get("provenance") or {}
        ds_l = (prov.get("dataset") or "").lower()
        ep_id = prov.get("episode") or prov.get("sampled_subfolder")
        if ds_l and ep_id:
            needed[ds_l].add(ep_id)
    print(f"  [prescan] L{level} {split}: {sum(len(v) for v in needed.values())} unique episodes needed "
          f"({dict((k, len(v)) for k, v in needed.items())})", flush=True)
    ep_cache: "OrderedDict[Tuple[str, str], Optional[List[Dict[str, Any]]]]" = OrderedDict(_bulk_prefetch(needed))
    print(f"  [prescan] loaded {len(ep_cache)} episodes into cache", flush=True)

    fh_inc = open(incremental_path, "w", encoding="utf-8") if incremental_path else None
    try:
        for it in src:
            new_it = it  # default: keep original
            if not is_impossible(it):
                stats["kept"] += 1
            else:
                stats["candidate"] += 1
                prov = it.get("provenance") or {}
                ds = (prov.get("dataset") or "").lower()
                ep_id = prov.get("episode") or prov.get("sampled_subfolder")
                if not ep_id:
                    stats["no_ep"] += 1
                else:
                    key = (ds, ep_id)
                    if key not in ep_cache:
                        try:
                            ep_cache[key] = load_episode(ds, ep_id)
                        except Exception as e:
                            ep_cache[key] = None
                            if stats["load_fail"] < 3:
                                print(f"    [load fail] {ds}/{ep_id[:20]}: {type(e).__name__}: {e}", flush=True)
                            stats["load_fail"] += 1
                        # LRU cap
                        while len(ep_cache) > ep_cache_max:
                            ep_cache.popitem(last=False)
                            gc.collect()
                    else:
                        ep_cache.move_to_end(key)
                    rows = ep_cache[key]
                    if not rows:
                        stats["src_missing"] += 1
                    else:
                        if level == 1: new = regen_l1_t7(it, rows)
                        elif level == 2: new = regen_l2_predict(it, rows)
                        elif level == 3: new = regen_l3(it, rows)
                        else: new = None
                        if new is None:
                            stats["regen_failed"] += 1
                            if stats["regen_failed"] <= 3:
                                _debug_fail(it, rows, level)
                        else:
                            new_it = new
                            stats["regen_ok"] += 1
            out.append(new_it)
            if fh_inc is not None:
                fh_inc.write(json.dumps(new_it) + "\n")
                if len(out) % flush_every == 0:
                    fh_inc.flush()
            if progress_every and stats["candidate"] and stats["candidate"] % progress_every == 0:
                print(f"    [progress] L{level} {split}: candidates={stats['candidate']} ok={stats['regen_ok']} "
                      f"fail={stats['regen_failed']} src_missing={stats['src_missing']} ep_cache={len(ep_cache)}", flush=True)
            if limit and (stats["regen_ok"] + stats["regen_failed"]) >= limit: break
    finally:
        if fh_inc is not None:
            fh_inc.flush()   # final tail (< flush_every rows)
            fh_inc.close()
    if incremental_path is not None:
        Path(str(incremental_path) + ".done").touch()
    return out, dict(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip any level_<L>_<split>.jsonl that already has a .done marker")
    ap.add_argument("--ep-cache-max", type=int, default=200,
                    help="cap on the resampled-episode cache size to bound RAM (default: 200)")
    ap.add_argument("--out", type=Path, default=Path("output/regen_10hz"))
    args = ap.parse_args()

    hf = _HF()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        for lvl in [1, 2, 3]:
            items, stats = regen_split(lvl, "test", hf, limit=5)
            print(f"L{lvl} test smoke: {stats}", flush=True)
        return 0

    if not args.full:
        print("nothing to do; pass --smoke or --full"); return 0

    for lvl in [1, 2, 3]:
        for split in ["train", "validation", "test"]:
            outp = args.out / f"level_{lvl}_{split}.jsonl"
            done_marker = outp.with_suffix(outp.suffix + ".done")
            if args.resume and done_marker.exists():
                print(f"L{lvl} {split}: SKIP (already done: {outp})", flush=True)
                continue
            outp.parent.mkdir(parents=True, exist_ok=True)
            items, stats = regen_split(lvl, split, hf, incremental_path=outp,
                                        ep_cache_max=args.ep_cache_max)
            print(f"L{lvl} {split}: {stats} -> {outp}", flush=True)
            # aggressive gc between splits
            import gc as _gc; _gc.collect()
            if args.push:
                remote = f"factorybench_qa/level_{lvl}/{split}.jsonl"
                hf.push_file(str(outp), remote, f"regen: fix impossible-at-10Hz windows (L{lvl}/{split})")
                print(f"  pushed hf://{remote}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
