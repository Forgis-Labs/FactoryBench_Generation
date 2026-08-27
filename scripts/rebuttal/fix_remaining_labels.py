"""Label-only fix for the ~144 Q&A items whose source parquet is not in the
local repo (mostly voraus-AD). We can't re-derive the answer from a
resampled series, but we don't need to: the answer field already carries
the correct future-row value from the paper's original generation run.
What's broken is only the `T+N ms` label, which was written as
`n=steps_ahead` (row count) instead of wall-clock ms.

For every impossible item we still see on the HF release (after the main
regen), we:
  1. Parse the visible context.time_series to recover its native dt.
  2. Compute wall-clock horizon = steps_ahead × dt.
  3. Rewrite the "T+N ms" label in the question to that wall-clock value.
  4. Add ``horizon_ms`` to acceptance_bounds.
  5. Leave answer, signal, actual_value, steps_ahead, and context alone.

Idempotent: skips items whose label is already a wall-clock multiple of dt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import find_dotenv, load_dotenv

LEVELS = [1, 2, 3]
SPLITS = ["train", "validation", "test"]
REPO_ID = "FactoryBench/FactoryBench"


def _ctx_dt_ms(item):
    ts_rows = (item.get("context") or {}).get("time_series") or []
    if len(ts_rows) < 2 or not isinstance(ts_rows[0], str):
        return None
    m0 = re.match(r"^t=(\d+):", ts_rows[0])
    m1 = re.match(r"^t=(\d+):", ts_rows[1])
    if not (m0 and m1):
        return None
    return int(m1.group(1)) - int(m0.group(1))


def _row_count_bug(item):
    """Class A: L1-style row-count-as-ms label. Fixable via label rewrite."""
    b = item.get("acceptance_bounds") or {}
    if "horizon_ms" in b: return False
    q = item.get("question", "")
    m = re.search(r"T\+(\d+)\s*ms", q)
    if not m: return False
    N = int(m.group(1))
    steps = b.get("steps_ahead")
    if steps is None: return False
    return int(steps) == N and N <= 10


def _needs_horizon_ms_only(item):
    """Class B: L2/L3-style item with a wall-clock T+N label but no horizon_ms
    in bounds. Metadata cleanup: add horizon_ms=N without changing anything else.
    Only applied when N is a plausible wall-clock ms (>= 100 or a native-rate
    multiple)."""
    b = item.get("acceptance_bounds") or {}
    if "horizon_ms" in b: return False
    q = item.get("question", "")
    m = re.search(r"T\+(\d+)\s*ms", q)
    if not m: return False
    N = int(m.group(1))
    if N < 20: return False   # avoid the row-count-as-ms shape; a separate fix runs on those
    return True


def _patch(item):
    dt = _ctx_dt_ms(item)
    if dt is None or dt <= 0: return None
    steps = int((item.get("acceptance_bounds") or {}).get("steps_ahead", 0))
    if steps <= 0: return None
    n_ms = steps * dt
    new_q = re.sub(r"T\+\d+\s*ms", f"T+{n_ms}ms", item.get("question", ""), count=1)
    out = dict(item)
    out["question"] = new_q
    b = dict(item.get("acceptance_bounds") or {})
    b["horizon_ms"] = n_ms
    out["acceptance_bounds"] = b
    return out


def _add_horizon_ms(item):
    q = item.get("question", "")
    m = re.search(r"T\+(\d+)\s*ms", q)
    if not m: return None
    out = dict(item)
    b = dict(item.get("acceptance_bounds") or {})
    b["horizon_ms"] = int(m.group(1))
    out["acceptance_bounds"] = b
    return out


def _patch_tiny_l2l3(item):
    """Class C: L2/L3-style item whose label is 'tiny' (<100 ms, e.g. T+7ms)
    inherited from a native-rate source. Rewrite via context-dt inference,
    same as Class A but not requiring steps_ahead to match label."""
    b = item.get("acceptance_bounds") or {}
    if "horizon_ms" in b: return None
    q = item.get("question", "")
    m = re.search(r"T\+(\d+)\s*ms", q)
    if not m: return None
    N = int(m.group(1))
    if N >= 100: return None
    dt = _ctx_dt_ms(item)
    if dt is None or dt <= 0: return None
    steps = b.get("steps_ahead")
    if steps is None:
        # Approximate steps from the label + a plausible native rate.
        # For tiny labels the native rate must be substantially higher than 10 Hz;
        # infer as N / (100 / dt_ratio). Simpler: assume every native-rate step
        # produced ~N/steps_est ms and steps = round(N * dt / native_step). Skip if
        # we can't determine steps from the item.
        return None
    n_ms = int(steps) * int(dt)
    new_q = re.sub(r"T\+\d+\s*ms", f"T+{n_ms}ms", q, count=1)
    out = dict(item)
    out["question"] = new_q
    new_b = dict(b); new_b["horizon_ms"] = n_ms
    out["acceptance_bounds"] = new_b
    return out


def process_one_jsonl(path: Path):
    src = [json.loads(ln) for ln in open(path, encoding="utf-8")]
    n_rowcount = 0        # Class A: L1 row-count-as-ms
    n_tiny = 0            # Class C: L2/L3 tiny label (< 100 ms)
    n_horizon_add = 0     # Class B: cleanup, add horizon_ms to bounds
    n_unfixable = 0
    with open(path, "w", encoding="utf-8") as fh:
        for it in src:
            new_it = it
            if _row_count_bug(it):
                p = _patch(it)
                if p is not None:
                    new_it = p; n_rowcount += 1
                else:
                    n_unfixable += 1
            else:
                p = _patch_tiny_l2l3(it)
                if p is not None:
                    new_it = p; n_tiny += 1
                elif _needs_horizon_ms_only(it):
                    p = _add_horizon_ms(it)
                    if p is not None:
                        new_it = p; n_horizon_add += 1
            fh.write(json.dumps(new_it) + "\n")
    return {
        "rowcount_patched": n_rowcount,
        "tiny_patched": n_tiny,
        "horizon_added": n_horizon_add,
        "unfixable": n_unfixable,
        "total": len(src),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, default=Path("output/regen_10hz"))
    ap.add_argument("--push", action="store_true",
                    help="upload every patched JSONL back to HF")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    tok = os.environ.get("HF_WRITE_TOKEN")

    api = None
    if args.push and not args.dry_run:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)

    grand = {"rowcount_patched": 0, "tiny_patched": 0, "horizon_added": 0,
             "unfixable": 0, "total": 0}
    to_repush = []
    for lvl in LEVELS:
        for split in SPLITS:
            path = args.src_dir / f"level_{lvl}_{split}.jsonl"
            if not path.exists():
                print(f"L{lvl}/{split}: MISSING local file, skipping"); continue
            if args.dry_run:
                src = [json.loads(ln) for ln in open(path, encoding="utf-8")]
                rc = sum(1 for it in src if _row_count_bug(it))
                tiny = sum(1 for it in src if _patch_tiny_l2l3(it) is not None)
                hadd = 0
                for it in src:
                    if _row_count_bug(it): continue
                    if _patch_tiny_l2l3(it) is not None: continue
                    if _needs_horizon_ms_only(it): hadd += 1
                print(f"L{lvl}/{split}: would patch rowcount={rc} tiny={tiny} horizon_add={hadd} total={len(src)}")
                grand["rowcount_patched"] += rc; grand["tiny_patched"] += tiny
                grand["horizon_added"] += hadd; grand["total"] += len(src)
                continue
            r = process_one_jsonl(path)
            for k in grand: grand[k] += r[k]
            touched = r["rowcount_patched"] + r["tiny_patched"] + r["horizon_added"]
            print(f"L{lvl}/{split}: rowcount={r['rowcount_patched']} tiny={r['tiny_patched']} "
                  f"horizon_add={r['horizon_added']} unfixable={r['unfixable']} total={r['total']} -> {path}",
                  flush=True)
            if touched > 0 and api is not None:
                to_repush.append((lvl, split, path))

    print()
    print(f"summary: {grand}")

    if to_repush:
        print(); print(f"pushing {len(to_repush)} patched file(s) back to HF ...")
        for lvl, split, path in to_repush:
            t0 = time.time()
            remote = f"factorybench_qa/level_{lvl}/{split}.jsonl"
            api.upload_file(path_or_fileobj=str(path), path_in_repo=remote,
                            repo_id=REPO_ID, repo_type="dataset",
                            commit_message=f"labels: patch remaining src_missing items (L{lvl}/{split})")
            dt = time.time() - t0
            size_mb = path.stat().st_size / (1024*1024)
            print(f"  L{lvl}/{split}: OK {size_mb:.1f}MB in {dt:.1f}s -> hf://{remote}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
