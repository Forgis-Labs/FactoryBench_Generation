"""Rewrite L1 template-7 questions with the correct millisecond horizon.

Root cause (reviewer W.3): the L1 template-7 generator called
`fill(..., n=steps_ahead)`, so the emitted question rendered as
`T+{steps_ahead}ms` — for example `T+4ms`. At the 10 Hz resampled
sensor rate one row is ~100 ms, so `steps_ahead=4` should read
`T+400ms`. L2/L3 templates were already correct because they passed
`n=n_ms` computed from the source timestamps.

This script walks a questions directory, finds every L1 template-7 item,
recomputes the correct `n_ms` from the actual timestamp axis stored in
the question's `context.time_series` block, rewrites the `question`
string in place, and archives the original alongside for review.

Nothing about the ground-truth answer or `steps_ahead` changes; only the
prompt string a model sees changes. So a rerun of the eval on ONLY the
patched items is enough to update the paper's L1 numbers.

Outputs:
  * `<out-dir>/questions_patched/level1/<id>.json`   — the fixed items
  * `<out-dir>/questions_original/level1/<id>.json`  — untouched copies
  * `<out-dir>/l1_t7_manifest.json`                  — {id, old_horizon,
                                                        new_horizon,
                                                        old_question, new_question}
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


TPLUS_RE = re.compile(r"T\+\d+ms")


def _timestamp_delta(q: dict) -> int | None:
    """Compute the correct T+ horizon in ms from the question's timestamps."""
    ctx = q.get("context") or {}
    ts_rows = ctx.get("time_series") or []
    if not ts_rows:
        return None
    # Rows look like "t=0: fp0=..., ..." — parse the leading t value.
    ts_vals: list[int] = []
    for row in ts_rows:
        if not isinstance(row, str):
            continue
        m = re.match(r"\s*t=(\d+)\s*:", row)
        if m:
            ts_vals.append(int(m.group(1)))
    if len(ts_vals) < 2:
        return None
    steps_ahead = int((q.get("acceptance_bounds") or {}).get("steps_ahead", 0))
    if steps_ahead <= 0:
        return None
    # Median inter-row interval is a robust estimate of the sample period.
    diffs = [b - a for a, b in zip(ts_vals[:-1], ts_vals[1:]) if b > a]
    if not diffs:
        return None
    diffs.sort()
    period_ms = diffs[len(diffs) // 2]
    return int(steps_ahead * period_ms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-root", default="output/test_eval/questions/level1")
    ap.add_argument("--out-dir", default="output/l1_t7_patched")
    ap.add_argument("--template-id", type=int, default=7)
    args = ap.parse_args()

    src = Path(args.questions_root)
    out = Path(args.out_dir)
    (out / "questions_patched" / "level1").mkdir(parents=True, exist_ok=True)
    (out / "questions_original" / "level1").mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    n_scanned = n_matched = n_patched = n_no_change = n_skipped = 0

    for qp in sorted(src.glob("*.json")):
        n_scanned += 1
        try:
            with open(qp, encoding="utf-8") as fh:
                q = json.load(fh)
        except Exception as e:
            print(f"[skip] {qp.name}: {e}", file=sys.stderr)
            n_skipped += 1
            continue
        if int(q.get("template_id", -1)) != args.template_id:
            continue
        n_matched += 1

        old_question = q.get("question", "")
        # Extract the currently-rendered horizon
        m_old = TPLUS_RE.search(old_question)
        old_horizon_str = m_old.group(0) if m_old else None
        old_horizon_ms = int(re.search(r"\d+", old_horizon_str).group()) if old_horizon_str else None

        new_horizon_ms = _timestamp_delta(q)
        if new_horizon_ms is None:
            n_skipped += 1
            continue

        if old_horizon_ms == new_horizon_ms:
            n_no_change += 1
            continue

        new_question = TPLUS_RE.sub(f"T+{new_horizon_ms}ms", old_question, count=1)
        q_patched = dict(q)
        q_patched["question"] = new_question
        ab = dict(q_patched.get("acceptance_bounds") or {})
        ab["horizon_ms"] = new_horizon_ms
        ab["horizon_ms_before_fix"] = old_horizon_ms
        q_patched["acceptance_bounds"] = ab

        shutil.copyfile(qp, out / "questions_original" / "level1" / qp.name)
        with open(out / "questions_patched" / "level1" / qp.name, "w", encoding="utf-8") as fh:
            json.dump(q_patched, fh, indent=2)

        manifest.append({
            "id": q.get("id") or qp.stem,
            "file": qp.name,
            "steps_ahead": (q.get("acceptance_bounds") or {}).get("steps_ahead"),
            "old_horizon_ms": old_horizon_ms,
            "new_horizon_ms": new_horizon_ms,
            "old_question": old_question,
            "new_question": new_question,
        })
        n_patched += 1

    with open(out / "l1_t7_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "template_id": args.template_id,
                "n_scanned": n_scanned,
                "n_template_matches": n_matched,
                "n_patched": n_patched,
                "n_no_change_needed": n_no_change,
                "n_skipped": n_skipped,
                "items": manifest,
            },
            fh,
            indent=2,
        )

    print(f"scanned {n_scanned} L1 items")
    print(f"  template {args.template_id}: {n_matched}")
    print(f"  patched: {n_patched}")
    print(f"  no change needed: {n_no_change}")
    print(f"  skipped (missing data): {n_skipped}")
    print(f"manifest -> {out / 'l1_t7_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
