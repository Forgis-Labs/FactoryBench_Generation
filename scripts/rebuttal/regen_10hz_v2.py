"""Regenerate every impossible-at-10Hz Q&A item using locally-available
10 Hz-resampled sources.

Local sources used (already at strict 10 Hz):
  data/factorywave/data/ur_signals_10hz/data.parquet          (UR3 pnp+peg)
  data/factorywave/data/ur_screwdriver_10hz/data.parquet      (UR3 screwing)
  data/factorywave/data/kuka_signals.parquet                  (KUKA; resampled here)
  data/open_datasets/aursad/experiment_<n>.parquet            (already ~10 Hz)
  data/open_datasets/vorausad/<episode>.parquet               (resampled here)

For each Q&A item with an impossible T+N label:
  1. Load its source episode from the corresponding local parquet (all rows)
  2. Snap timestamps to a strict 10-ms lattice if not already
  3. Slice subseries[start:start+len] from the resampled rows
  4. Pull future_val at row `start + len + steps_ahead - 1`
  5. Rebuild question text with new wall-clock T+N ms, rewrite answer,
     acceptance bounds, and context.time_series
  6. Preserve UUID

Only the Q&A JSONL files are updated. Model replies and panel scores are
untouched.
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

DATA_DIR = Path("data")
TARGET_DT_MS = 100

# Source parquet layout (all local)
FW_LOCAL_PARQUETS = {
    "ur":            DATA_DIR / "factorywave/data/ur_signals_10hz/data.parquet",
    "ur_screwdriver": DATA_DIR / "factorywave/data/ur_screwdriver_10hz/data.parquet",
    "kuka":          DATA_DIR / "factorywave/data/kuka_signals.parquet",
}
AURSAD_DIR = DATA_DIR / "open_datasets/aursad"
VORAUS_DIR = DATA_DIR / "open_datasets/vorausad"

# In-memory episode caches
_FW_DF: Dict[str, pd.DataFrame] = {}
_FW_EP_INDEX: Dict[str, set] = {}


def _load_fw(track: str) -> pd.DataFrame:
    if track in _FW_DF: return _FW_DF[track]
    print(f"  loading {FW_LOCAL_PARQUETS[track]} ...", flush=True)
    df = pd.read_parquet(FW_LOCAL_PARQUETS[track])
    _FW_DF[track] = df
    _FW_EP_INDEX[track] = set(df["episode_id"].unique())
    return df


def _fw_episode_rows(episode_id: str) -> Optional[List[Dict[str, Any]]]:
    """Return rows for a FactoryWave episode from local 10 Hz parquet."""
    for track in FW_LOCAL_PARQUETS:
        _load_fw(track)
        if episode_id in _FW_EP_INDEX[track]:
            sub = _FW_DF[track][_FW_DF[track]["episode_id"] == episode_id].sort_values("time")
            return _df_to_rows(sub, native_snap=(track == "kuka"))
    return None


def _df_to_rows(df: pd.DataFrame, native_snap: bool = False) -> List[Dict[str, Any]]:
    if "time" not in df.columns: return []
    ts_col = df["time"]
    if pd.api.types.is_datetime64_any_dtype(ts_col):
        # drop tz if present, then cast to int64 ns
        if getattr(ts_col.dt, "tz", None) is not None:
            ts_col = ts_col.dt.tz_convert("UTC").dt.tz_localize(None)
        ts_ns = ts_col.astype("datetime64[ns]").astype(np.int64).to_numpy()
        ts_ms = (ts_ns - ts_ns[0]) // 1_000_000
    else:
        arr = np.asarray(ts_col.to_list(), dtype=np.int64)
        ts_ms = arr - arr[0]
    signal_cols = [c for c in df.columns if c not in ("time", "episode_id")]
    rows = []
    df_rec = df[signal_cols].reset_index(drop=True)
    for i in range(len(df)):
        r = {"timestamp_ms": int(ts_ms[i])}
        for c in signal_cols:
            v = df_rec.iloc[i][c]
            if pd.isna(v): r[c] = None
            elif isinstance(v, (int, float, np.integer, np.floating)): r[c] = float(v)
            else: r[c] = v
        rows.append(r)
    if native_snap:
        rows = _snap_to_10hz(rows)
    return rows


def _snap_to_10hz(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Snap rows to a strict 10-ms lattice. If native dt >= 90 ms already, no-op-ish."""
    if len(rows) < 2: return rows
    ts = np.array([r["timestamp_ms"] for r in rows], dtype=np.int64)
    dt_med = float(np.median(np.diff(ts)))
    if 90 <= dt_med <= 110:
        # already ~10 Hz: just rewrite timestamps to a strict lattice
        for i, r in enumerate(rows):
            r["timestamp_ms"] = i * TARGET_DT_MS
        return rows
    # native rate faster than 10 Hz: pick every ~(TARGET_DT_MS / dt_med)-th row
    stride = max(1, int(round(TARGET_DT_MS / dt_med)))
    out = rows[::stride]
    for i, r in enumerate(out):
        r["timestamp_ms"] = i * TARGET_DT_MS
    return out


def _aursad_episode_rows(episode_id: str) -> Optional[List[Dict[str, Any]]]:
    p = AURSAD_DIR / f"{episode_id}.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p)
    if "timestamp" in df.columns:
        ts_ms = df["timestamp"].astype(np.int64).to_numpy() - int(df["timestamp"].iloc[0])
    else:
        ts_ms = np.arange(len(df)) * 100
    sig_cols = [c for c in df.columns if c not in ("timestamp", "sample_nr")]
    rows = []
    for i in range(len(df)):
        row = {"timestamp_ms": int(ts_ms[i])}
        for c in sig_cols:
            v = df.iloc[i][c]
            row[c] = None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
        rows.append(row)
    return _snap_to_10hz(rows)


def _voraus_episode_rows(episode_id: str) -> Optional[List[Dict[str, Any]]]:
    p = VORAUS_DIR / f"{episode_id}.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p)
    tcol = "timestamp_ms" if "timestamp_ms" in df.columns else "time"
    if tcol in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[tcol]):
            ts_ns = df[tcol].astype("datetime64[ns]").astype(np.int64).to_numpy()
            ts_ms = (ts_ns - ts_ns[0]) // 1_000_000
        else:
            arr = np.asarray(df[tcol].to_list(), dtype=np.int64)
            ts_ms = arr - arr[0]
    else:
        ts_ms = np.arange(len(df)) * 20
    sig_cols = [c for c in df.columns if c != tcol]
    rows = []
    for i in range(len(df)):
        row = {"timestamp_ms": int(ts_ms[i])}
        for c in sig_cols:
            v = df.iloc[i][c]
            row[c] = None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
        rows.append(row)
    return _snap_to_10hz(rows)


def load_episode(dataset: str, episode_id: str) -> Optional[List[Dict[str, Any]]]:
    ds = (dataset or "").lower()
    if ds == "factorywave":
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
}
def _short(name: str) -> str:
    for prefix, short in _ACRONYMS.items():
        if name.startswith(prefix):
            return short + name[len(prefix):]
    return name


def _rebuild_context(rows_10hz: List[Dict[str, Any]], start: int, length: int) -> Dict[str, Any]:
    sub = rows_10hz[start:start + length]
    if not sub: return {"time_series": []}
    lines = []
    for r in sub:
        ts = int(r.get("timestamp_ms", 0))
        parts = []
        for k, v in r.items():
            if k == "timestamp_ms": continue
            if v is None: continue
            if isinstance(v, (int, float)):
                parts.append(f"{_short(k)}={round(v, 4)}")
        lines.append(f"t={ts}: " + ", ".join(parts))
    return {"time_series": lines}


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
    out["context"] = _rebuild_context(rows_10hz, start, length)
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
    out["context"] = _rebuild_context(rows_10hz, start, length)
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
    if level == 1: return True   # L1 label-count-as-ms bug affects all L1.7
    if N % 100 != 0: return True
    return False


def _debug_fail(item, rows, level):
    prov = item.get("provenance") or {}
    bounds = item.get("acceptance_bounds") or {}
    start = prov.get("subseries_start_index")
    length = prov.get("subseries_length")
    signal = bounds.get("signal")
    steps = bounds.get("steps_ahead")
    fut = None
    try:
        fut = int(start) + int(length) + int(steps or 1) - 1
    except Exception:
        pass
    print(f"    [dbg L{level}] id={item.get('id')[:8]} ds={prov.get('dataset')} "
          f"ep={(prov.get('episode') or prov.get('sampled_subfolder','') or '')[:20]} "
          f"start={start} len={length} steps={steps} future_idx={fut} "
          f"rows_10hz={len(rows) if rows else 0} sig={signal} "
          f"sig_in_row0={signal in rows[0] if rows and signal else '?'}")


def regen_split(level: int, split: str, hf: _HF, limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = defaultdict(int)
    path = hf.dl(f"factorybench_qa/level_{level}/{split}.jsonl")
    src = [json.loads(ln) for ln in open(path, encoding="utf-8")]
    out = []
    ep_cache: Dict[Tuple[str, str], Optional[List[Dict[str, Any]]]] = {}
    for it in src:
        if not is_impossible(it):
            out.append(it); stats["kept"] += 1; continue
        stats["candidate"] += 1
        prov = it.get("provenance") or {}
        ds = (prov.get("dataset") or "").lower()
        ep_id = prov.get("episode") or prov.get("sampled_subfolder")
        if not ep_id: out.append(it); stats["no_ep"] += 1; continue
        key = (ds, ep_id)
        if key not in ep_cache:
            ep_cache[key] = load_episode(ds, ep_id)
        rows = ep_cache[key]
        if not rows: out.append(it); stats["src_missing"] += 1; continue
        if level == 1: new = regen_l1_t7(it, rows)
        elif level == 2: new = regen_l2_predict(it, rows)
        elif level == 3: new = regen_l3(it, rows)
        else: new = None
        if new is None:
            out.append(it); stats["regen_failed"] += 1
            if stats["regen_failed"] <= 3:
                _debug_fail(it, rows, level)
        else: out.append(new); stats["regen_ok"] += 1
        if limit and (stats["regen_ok"] + stats["regen_failed"]) >= limit: break
    return out, dict(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--push", action="store_true", help="after --full, push updated JSONLs to HF")
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
            items, stats = regen_split(lvl, split, hf)
            outp = args.out / f"level_{lvl}_{split}.jsonl"
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", encoding="utf-8") as fh:
                for it in items:
                    fh.write(json.dumps(it) + "\n")
            print(f"L{lvl} {split}: {stats} -> {outp}", flush=True)
            if args.push:
                remote = f"factorybench_qa/level_{lvl}/{split}.jsonl"
                hf.push_file(str(outp), remote, f"regen: fix impossible-at-10Hz windows (L{lvl}/{split})")
                print(f"  pushed to hf://{remote}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
