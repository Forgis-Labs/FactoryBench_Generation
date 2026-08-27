"""Add per-episode `margin` to L1 template-7 acceptance_bounds.

Template-7 currently ships bounds of the form
    {"signal": "feedback_pos_0", "steps_ahead": 7, "actual_value": 20.8252}
with no `margin`/`min`/`max`. The numerical scorer then falls back to
`abs(pred - gt) < 1e-4`, which no model can hit for a joint position, so
everyone scores 0 on all 364 template-7 items in the L1 test set.

The paper's Chronos-Bolt appendix (App. B) reports 51.5% signed CC on
this template family and describes the scoring as "the same
chance-corrected piecewise-margin rule used in the main paper
(Appendix E; E=1/4)". Under the tensor branch's calibration, this
means margin = R/12 where R is the natural range of the channel.

We use the per-episode observed range (max - min of the target signal
across the visible window in `context.time_series`), which is what a
model actually has to extrapolate from. That gives E=1/4 chance under
uniform-random-in-range guessing while keeping the margin meaningful
per item (a static ~30° for all joints is far too loose).

Idempotent: skips items that already carry a `margin`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROW_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)=([\-0-9.eE+]+)")

# Long-name -> acronym seen in question prompts (matches sims/*/pick_and_place)
ACRONYMS = {
    "effort_target_torque_":   "ett",
    "effort_current_feedback_": "ecf",
    "est_contact_force_":       "ecf",
    "feedback_pos_":           "fp",
    "feedback_speed_":         "fs",
    "setpoint_pos_":           "sp",
    "timestamp_ms":            "tm",
}


def _acronym(long_name: str) -> str:
    for prefix, short in ACRONYMS.items():
        if long_name.startswith(prefix):
            return short + long_name[len(prefix):]
    return long_name


def _extract_signal_values(rows: List[Any], signal: str) -> List[float]:
    """Pull the target signal from the question's time-series rows.

    Rows may be dicts (already channel-keyed) or strings of the form
    ``t=<ts>: k1=v1, k2=v2, ...`` where the keys use acronyms.
    """
    if not rows:
        return []
    if isinstance(rows[0], dict):
        vals = []
        for r in rows:
            if signal in r:
                try: vals.append(float(r[signal]))
                except Exception: pass
        return vals
    # string rows use acronyms
    short = _acronym(signal)
    vals = []
    for row in rows:
        if not isinstance(row, str): continue
        for m in _ROW_RE.finditer(row):
            k, v = m.group(1), m.group(2)
            if k == short:
                try: vals.append(float(v))
                except ValueError: pass
                break
    return vals


def patch_one(question_path: Path) -> str:
    q = json.loads(question_path.read_text(encoding="utf-8"))
    if q.get("template_type") != "predictive" or q.get("template_id") != 7:
        return "skip: not template-7"
    ab = q.get("acceptance_bounds") or {}
    if "margin" in ab or "min" in ab:
        return "skip: already has margin/min"
    signal = ab.get("signal")
    if not signal:
        return "skip: no signal"
    ts_rows = (q.get("context") or {}).get("time_series") or []
    vals = _extract_signal_values(ts_rows, signal)
    if len(vals) < 2:
        return f"skip: only {len(vals)} values for {signal!r}"
    r = max(vals) - min(vals)
    if r <= 0:
        # flat signal: fall back to a small absolute margin so the item is scoreable
        margin = 0.05
    else:
        margin = r / 12.0
    ab = dict(ab)
    ab["margin"] = round(float(margin), 4)
    ab["_margin_source"] = "signal_range_over_12"
    q["acceptance_bounds"] = ab
    question_path.write_text(json.dumps(q, indent=2), encoding="utf-8")
    return f"ok: margin={ab['margin']} (range={r:.3f} for {signal})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-root", type=Path,
                    default=Path("output/test_eval/questions/level1"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(args.questions_root.glob("*.json"))
    print(f"scanning {len(files)} L1 question files...")
    stats: Dict[str, int] = {"ok": 0, "skip": 0, "err": 0}
    for f in files:
        try:
            if args.dry_run:
                q = json.loads(f.read_text(encoding="utf-8"))
                if q.get("template_id") == 7 and q.get("template_type") == "predictive":
                    ab = q.get("acceptance_bounds") or {}
                    if "margin" not in ab and "min" not in ab:
                        stats["ok"] += 1
                    else:
                        stats["skip"] += 1
            else:
                res = patch_one(f)
                if res.startswith("ok"): stats["ok"] += 1
                else:                    stats["skip"] += 1
        except Exception as e:
            stats["err"] += 1
            print(f"  ERR {f.name}: {e}")
    print(f"patched: {stats['ok']}  skipped: {stats['skip']}  errors: {stats['err']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
