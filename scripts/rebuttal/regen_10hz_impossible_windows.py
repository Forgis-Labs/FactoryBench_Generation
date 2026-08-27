"""Regenerate every Q&A item with an impossible-at-10Hz window using a
proper 10Hz-resampled source.

Scope:
  - L1 template-7 (prediction) items where the "T+Nms" label is a row-count
    not a wall-clock ms (~3,700 items).
  - L2 template-4 / template-5 items where the T+N label is real wall-clock
    ms but on a non-10Hz source, producing non-multiple-of-100 horizons
    (~1,350 items).
  - L3 template-4 items with the same issue as L2 (~1,050 items).

Fix strategy per item:
  1. Load the item's source episode from its source parquet (ur_signals,
     kuka_signals, ur_screwdriver_signals for FactoryWave; local files for
     AURSAD / voraus-AD).
  2. Anti-alias filter + resample the episode to strict 10 Hz.
  3. Rebuild the visible context (subseries[start:start+len]) from the
     resampled series.
  4. Compute new steps_ahead as the smallest integer in [1,10] and pull the
     future row from the resampled series. Recompute the ground-truth
     answer and acceptance-bound margin from the resampled window.
  5. Rewrite the question text with the correct wall-clock ms (N × 100).
  6. Emit the corrected item (preserving UUID).

Preserves UUIDs so downstream references still work. Does NOT touch model
replies or panel scores (that's a separate step handled elsewhere).

Usage:
    python scripts/rebuttal/regen_10hz_impossible_windows.py --smoke      # 5 items each L1/L2/L3
    python scripts/rebuttal/regen_10hz_impossible_windows.py --full       # all 6,094 items
    python scripts/rebuttal/regen_10hz_impossible_windows.py --push       # push updated JSONLs to HF
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

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv


# ---------------- constants ----------------
HF_REPO = "FactoryBench/FactoryBench"
FACTORYWAVE_PARQUETS = {
    "ur":            "factorywave/ur_signals.parquet",
    "ur_screwdriver": "factorywave/ur_screwdriver_signals.parquet",
    "kuka":          "factorywave/kuka_signals.parquet",
}
AURSAD_DIR = Path("data/open_datasets/aursad")
VORAUS_DIR = Path("data/open_datasets/vorausad")

TARGET_HZ = 10.0
TARGET_DT_MS = int(round(1000.0 / TARGET_HZ))   # 100 ms


# ---------------- resampling ----------------
def resample_to_10hz(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Anti-alias filter + decimate rows to strict 10 Hz.

    `rows` is a list of dicts with a `timestamp_ms` key and per-signal
    scalar values. Returns a new list of rows on a strict 10-ms lattice
    starting at rows[0]['timestamp_ms'].
    """
    if not rows: return []
    ts = np.array([r.get("timestamp_ms", i) for i, r in enumerate(rows)], dtype=float)
    if len(ts) < 2: return list(rows)
    # signal columns: everything that's numeric and not timestamp_ms
    cols = [k for k in rows[0].keys() if k != "timestamp_ms" and isinstance(rows[0].get(k), (int, float))]
    if not cols: return list(rows)
    # native dt
    dts = np.diff(ts)
    dt_med = float(np.median(dts))
    if dt_med >= TARGET_DT_MS - 5:
        # already at ~10Hz or lower; just snap timestamps to lattice
        return _snap_to_lattice(rows, cols, ts)
    # anti-alias + decimate: 3rd-order Butterworth at 0.4 * target_rate
    from scipy.signal import butter, filtfilt
    native_rate = 1000.0 / dt_med
    nyq_new = TARGET_HZ / 2.0
    cutoff = 0.4 * TARGET_HZ  # a bit below nyquist to leave headroom
    b, a = butter(3, cutoff / (native_rate / 2.0), btype="low")
    ts_out = np.arange(ts[0], ts[-1] + 1e-6, TARGET_DT_MS)
    out_rows = []
    for j, t in enumerate(ts_out):
        row_out = {"timestamp_ms": int(t)}
        for c in cols:
            v = np.array([r.get(c, np.nan) for r in rows], dtype=float)
            if not np.all(np.isnan(v)):
                try:
                    v_filt = filtfilt(b, a, v)
                    row_out[c] = float(np.interp(t, ts, v_filt))
                except Exception:
                    row_out[c] = float(np.interp(t, ts, v))
        # also preserve non-numeric labels from the nearest source row
        j_src = int(np.clip(np.argmin(np.abs(ts - t)), 0, len(rows) - 1))
        for k, val in rows[j_src].items():
            if k not in row_out and not isinstance(val, (int, float)):
                row_out[k] = val
        out_rows.append(row_out)
    return out_rows


def _snap_to_lattice(rows: List[Dict[str, Any]], cols: List[str], ts: np.ndarray) -> List[Dict[str, Any]]:
    """When native rate is already ~10Hz, snap timestamps to a strict lattice."""
    if len(ts) == 0: return []
    ts_out = np.arange(ts[0], ts[-1] + 1e-6, TARGET_DT_MS)
    out = []
    for t in ts_out:
        j = int(np.argmin(np.abs(ts - t)))
        row = {"timestamp_ms": int(t)}
        for c in cols:
            row[c] = rows[j].get(c)
        for k, val in rows[j].items():
            if k not in row and not isinstance(val, (int, float)):
                row[k] = val
        out.append(row)
    return out


# ---------------- source loaders ----------------
_PARQUET_CACHE: Dict[str, pd.DataFrame] = {}


_PARQUET_EPS_INDEX: Dict[str, set] = {}          # track -> set(episode_id)
_PARQUET_PATHS: Dict[str, str] = {}              # track -> local path


def _index_parquet_episodes(hf, track: str, filename: str) -> None:
    if track in _PARQUET_EPS_INDEX: return
    path = hf.download(filename)
    _PARQUET_PATHS[track] = path
    # read only the episode_id column to build the index without loading everything
    import pyarrow.parquet as pq
    tbl = pq.read_table(path, columns=["episode_id"])
    _PARQUET_EPS_INDEX[track] = set(tbl.column("episode_id").to_pylist())


def load_factorywave_episode(hf, episode_id: str) -> Optional[List[Dict[str, Any]]]:
    """Return the episode's rows (dicts) from whichever FactoryWave parquet contains it."""
    for track, filename in FACTORYWAVE_PARQUETS.items():
        _index_parquet_episodes(hf, track, filename)
        if episode_id not in _PARQUET_EPS_INDEX[track]:
            continue
        # push down the filter so we only materialise this one episode
        import pyarrow.parquet as pq
        tbl = pq.read_table(
            _PARQUET_PATHS[track],
            filters=[("episode_id", "=", episode_id)],
        )
        df = tbl.to_pandas().sort_values("time")
        return _parquet_rows_to_dicts(df)
    return None


def _parquet_rows_to_dicts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert dataframe with a `time` column into rows with timestamp_ms."""
    out = []
    if "time" not in df.columns:
        return out
    ts_col = df["time"]
    # Force to a proper datetime64 or int, avoiding Timestamp objects
    if pd.api.types.is_datetime64_any_dtype(ts_col):
        ts_ns = ts_col.dt.tz_convert(None).astype("datetime64[ns]").astype(np.int64).to_numpy() \
            if ts_col.dt.tz is not None else \
            ts_col.astype("datetime64[ns]").astype(np.int64).to_numpy()
        ts_ms = (ts_ns - ts_ns[0]) // 1_000_000
    else:
        arr = np.asarray(ts_col.to_list(), dtype=np.int64)
        ts_ms = arr - arr[0]
    signal_cols = [c for c in df.columns if c not in ("time", "episode_id")]
    for i in range(len(df)):
        row = {"timestamp_ms": int(ts_ms[i])}
        for c in signal_cols:
            v = df.iloc[i][c]
            if pd.isna(v):
                row[c] = None
            elif isinstance(v, (int, float, np.integer, np.floating)):
                row[c] = float(v)
            else:
                row[c] = v
        out.append(row)
    return out


def load_aursad_episode(episode_id: str) -> Optional[List[Dict[str, Any]]]:
    """AURSAD episodes are stored as data/open_datasets/aursad/experiment_<n>.parquet."""
    p = AURSAD_DIR / f"{episode_id}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "timestamp" in df.columns:
        ts_ms = df["timestamp"].astype(np.int64).to_numpy()
        ts_ms = ts_ms - ts_ms[0]
    else:
        ts_ms = np.arange(len(df)) * 100
    sig_cols = [c for c in df.columns if c not in ("timestamp", "sample_nr")]
    rows = []
    for i, ts in enumerate(ts_ms):
        row = {"timestamp_ms": int(ts)}
        for c in sig_cols:
            v = df.iloc[i][c]
            row[c] = None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
        rows.append(row)
    return rows


def load_voraus_episode(episode_id: str) -> Optional[List[Dict[str, Any]]]:
    p = VORAUS_DIR / f"{episode_id}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    tcol = "timestamp_ms" if "timestamp_ms" in df.columns else "time"
    ts_ms = df[tcol].astype(np.int64).to_numpy() if tcol in df.columns else np.arange(len(df)) * 20
    ts_ms = ts_ms - ts_ms[0]
    sig_cols = [c for c in df.columns if c != tcol]
    rows = []
    for i, ts in enumerate(ts_ms):
        row = {"timestamp_ms": int(ts)}
        for c in sig_cols:
            v = df.iloc[i][c]
            row[c] = None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
        rows.append(row)
    return rows


# ---------------- item regeneration ----------------
def rebuild_context_ts(rows_10hz: List[Dict[str, Any]], start_idx: int, length: int,
                       important_features: Optional[List[str]] = None) -> Dict[str, Any]:
    """Rebuild `context.time_series` (list of 't=<ms>: k=v, ...' strings) from resampled rows."""
    sub = rows_10hz[start_idx:start_idx + length]
    if not sub: return {"time_series": []}
    lines = []
    for r in sub:
        ts = r.get("timestamp_ms", 0)
        parts = []
        for k, v in r.items():
            if k == "timestamp_ms": continue
            if important_features and k not in important_features: continue
            if v is None: continue
            if isinstance(v, (int, float)):
                parts.append(f"{_short_name(k)}={round(v, 4)}")
        lines.append(f"t={int(ts)}: " + ", ".join(parts))
    return {"time_series": lines}


_ACRONYMS = {
    "effort_target_torque_": "ett", "feedback_pos_": "fp", "setpoint_pos_": "sp",
    "feedback_speed_": "fs", "est_contact_force_": "ecf", "timestamp_ms": "tm",
}
def _short_name(name: str) -> str:
    for prefix, short in _ACRONYMS.items():
        if name.startswith(prefix):
            return short + name[len(prefix):]
    return name


def regen_l1_t7_item(item: Dict[str, Any], rows_10hz: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """L1 template-7: predict signal value N rows ahead (N ∈ [1, 10] → 100..1000 ms)."""
    prov = item.get("provenance") or {}
    start = int(prov.get("subseries_start_index", 0))
    length = int(prov.get("subseries_length", 0))
    bounds = dict(item.get("acceptance_bounds") or {})
    signal = bounds.get("signal")
    steps = int(bounds.get("steps_ahead", 1))
    if not signal or start < 0 or length <= 0:
        return None
    future_idx = start + length + steps - 1
    if future_idx >= len(rows_10hz) or start + length > len(rows_10hz):
        return None
    future_row = rows_10hz[future_idx]
    last_ctx = rows_10hz[start + length - 1]
    if signal not in future_row or future_row[signal] is None:
        return None
    n_ms = int(future_row["timestamp_ms"]) - int(last_ctx["timestamp_ms"])
    new_answer = round(float(future_row[signal]), 4)
    new_q = re.sub(r"T\+\d+\s*ms", f"T+{n_ms}ms", item.get("question", ""), count=1)
    new_bounds = {**bounds, "actual_value": new_answer, "horizon_ms": n_ms}
    # rebuild context if we can identify important features
    ctx = rebuild_context_ts(rows_10hz, start, length)
    out = dict(item)
    out["question"] = new_q
    out["answer"] = new_answer
    out["acceptance_bounds"] = new_bounds
    out["context"] = ctx
    return out


def regen_l2_predict_item(item: Dict[str, Any], rows_10hz: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """L2 template-4 (scalar) / template-5 (tensor) predict."""
    prov = item.get("provenance") or {}
    start = int(prov.get("subseries_start_index", 0))
    length = int(prov.get("subseries_length", 0))
    bounds = dict(item.get("acceptance_bounds") or {})
    if start < 0 or length <= 0:
        return None
    # For L2 the event marker matters. Simplest: use the last visible row as t_event, pick steps_ahead=1..10
    # so future is at N × 100 ms after the visible window ends.
    # We choose steps ahead equal to whatever the item originally used, clamped so future_idx exists.
    # If original steps_ahead not available (some items only have `horizon_ms`), derive N from horizon_ms.
    horizon_ms = int(bounds.get("horizon_ms", 0)) if bounds.get("horizon_ms") else None
    if not horizon_ms:
        # infer N from the question's T+N label
        m = re.search(r"T\+(\d+)\s*ms", item.get("question", ""))
        if not m: return None
        horizon_ms = int(m.group(1))
    steps_ahead = max(1, min(10, int(round(horizon_ms / TARGET_DT_MS))))
    future_idx = start + length + steps_ahead - 1
    if future_idx >= len(rows_10hz): return None
    future_row = rows_10hz[future_idx]
    last_ctx = rows_10hz[start + length - 1]
    n_ms = int(future_row["timestamp_ms"]) - int(last_ctx["timestamp_ms"])
    signal = bounds.get("signal")
    if signal is None or isinstance(bounds.get("std"), list):
        # tensor (t5): signal is a joint-base name; expand to signal_0..signal_5
        base = signal if isinstance(signal, str) else _guess_tensor_base(bounds)
        if base is None: return None
        vals = []
        for j in range(6):
            key = f"{base}_{j}"
            v = future_row.get(key)
            if v is None: return None
            vals.append(round(float(v), 6))
        new_answer = "[" + ",".join(str(v) for v in vals) + "]"
    else:
        v = future_row.get(signal)
        if v is None: return None
        new_answer = round(float(v), 6)
    new_q = re.sub(r"T\+\d+\s*ms", f"T+{n_ms}ms", item.get("question", ""), count=1)
    out = dict(item)
    out["question"] = new_q
    out["answer"] = new_answer
    new_bounds = dict(bounds)
    if isinstance(new_answer, str):
        # tensor: recompute per-joint std / margin from visible window
        base = signal if isinstance(signal, str) else _guess_tensor_base(bounds)
        stds = []
        for j in range(6):
            key = f"{base}_{j}"
            vals = [r.get(key) for r in rows_10hz[start:start + length] if r.get(key) is not None]
            stds.append(round(float(np.std(vals)) if vals else 0.0, 6))
        new_bounds["std"] = stds
        new_bounds["margin"] = [round(s * 0.75, 6) for s in stds]
    else:
        vals = [r.get(signal) for r in rows_10hz[start:start + length] if r.get(signal) is not None]
        std = float(np.std(vals)) if vals else 0.0
        new_bounds["std"] = round(std, 6)
        new_bounds["margin"] = round(std * 0.75, 6)
    new_bounds["horizon_ms"] = n_ms
    out["acceptance_bounds"] = new_bounds
    out["context"] = rebuild_context_ts(rows_10hz, start, length)
    return out


def _guess_tensor_base(bounds: Dict[str, Any]) -> Optional[str]:
    return bounds.get("signal") if isinstance(bounds.get("signal"), str) else None


# regen for L3 template-4: same shape as L2 t4 but on counterpart/counterfactual episode
def regen_l3_item(item: Dict[str, Any], rows_10hz: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return regen_l2_predict_item(item, rows_10hz)


# ---------------- HF wrapper ----------------
class _HF:
    def __init__(self):
        load_dotenv(find_dotenv(usecwd=True))
        from huggingface_hub import hf_hub_download
        self._hf = hf_hub_download
        self.tok = os.environ["HF_WRITE_TOKEN"]
    def download(self, filename: str) -> str:
        return self._hf(HF_REPO, filename, repo_type="dataset", token=self.tok)


# ---------------- main pipeline ----------------
def is_impossible(item: Dict[str, Any]) -> bool:
    m = re.search(r"T\+(\d+)\s*ms", item.get("question", ""))
    if not m: return False
    N = int(m.group(1))
    level = item.get("level")
    if level == 1: return True   # all L1 t7 items have the label bug
    if N % 100 != 0: return True
    return False


def _log_failure(item, rows_10hz, level):
    prov = item.get("provenance") or {}
    bounds = item.get("acceptance_bounds") or {}
    print(f"    [DBG L{level} regen fail] id={item.get('id')} ds={prov.get('dataset')} "
          f"ep={prov.get('episode') or prov.get('sampled_subfolder')} "
          f"start={prov.get('subseries_start_index')} len={prov.get('subseries_length')} "
          f"steps_ahead={bounds.get('steps_ahead')} signal={bounds.get('signal')} "
          f"rows_10hz_len={len(rows_10hz) if rows_10hz else 0}")


def regen_split(level: int, split: str, hf: _HF, limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Iterate through a split's JSONL, regen every impossible item. Return (items, stats)."""
    stats = defaultdict(int)
    path = hf.download(f"factorybench_qa/level_{level}/{split}.jsonl")
    src_items = [json.loads(ln) for ln in open(path, encoding="utf-8")]
    out_items = []
    ep_cache: Dict[Tuple[str, str], Optional[List[Dict[str, Any]]]] = {}
    for it in src_items:
        if not is_impossible(it):
            out_items.append(it); stats["kept"] += 1; continue
        stats["candidate"] += 1
        prov = it.get("provenance") or {}
        ds = (prov.get("dataset") or "").lower()
        ep_id = prov.get("episode") or prov.get("sampled_subfolder")
        if ep_id is None:
            out_items.append(it); stats["no_ep"] += 1; continue
        key = (ds, ep_id)
        if key not in ep_cache:
            if ds == "factorywave":
                raw = load_factorywave_episode(hf, ep_id)
            elif ds == "aursad":
                raw = load_aursad_episode(ep_id)
            elif "voraus" in ds:
                raw = load_voraus_episode(ep_id)
            else:
                raw = None
            if raw is None:
                ep_cache[key] = None
            else:
                ep_cache[key] = resample_to_10hz(raw)
        rows_10hz = ep_cache[key]
        if rows_10hz is None:
            out_items.append(it); stats["src_missing"] += 1; continue
        if level == 1:
            new = regen_l1_t7_item(it, rows_10hz)
        elif level == 2:
            new = regen_l2_predict_item(it, rows_10hz)
        elif level == 3:
            new = regen_l3_item(it, rows_10hz)
        else:
            new = None
        if new is None:
            out_items.append(it); stats["regen_failed"] += 1
            if stats["regen_failed"] <= 3:
                _log_failure(it, rows_10hz, level)
        else:
            out_items.append(new); stats["regen_ok"] += 1
        if limit and stats["regen_ok"] + stats["regen_failed"] >= limit:
            break
    return out_items, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run on 5 items per level, don't write output")
    ap.add_argument("--full",  action="store_true", help="regenerate every impossible item across all splits")
    ap.add_argument("--push",  action="store_true", help="after regenerating, push updated JSONLs to HF")
    ap.add_argument("--out",   type=Path, default=Path("output/regen_10hz"))
    args = ap.parse_args()

    hf = _HF()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        for lvl in [1, 2, 3]:
            items, stats = regen_split(lvl, "test", hf, limit=5)
            print(f"L{lvl}: {dict(stats)}")
        return 0

    if not args.full:
        print("nothing to do; pass --smoke or --full"); return 0

    for lvl in [1, 2, 3]:
        for split in ["train", "validation", "test"]:
            items, stats = regen_split(lvl, split, hf)
            outp = args.out / f"level_{lvl}_{split}.jsonl"
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", encoding="utf-8") as fh:
                for it in items:
                    fh.write(json.dumps(it) + "\n")
            print(f"L{lvl} {split}: {dict(stats)} -> {outp}")

    if args.push:
        # separate step; write a follow-up
        print("--push not yet wired; use scripts/rebuttal/push_regen_jsonls.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
