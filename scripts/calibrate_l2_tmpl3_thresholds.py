#!/usr/bin/env python3
"""Calibrate L2 template-3 option thresholds to a ~50/50 true/false split.

Level 2 template 3 ("The sensor stream below is from a robot exhibiting
{anomaly}. Select all statements that apply...") builds four multi-select
statements whose truth depends on whether a measured quantity crosses a
sampled threshold. Those thresholds are drawn from Gaussians centred on
``DEFAULT_THRESHOLDS``, and those centres were hand-picked rather than fitted
to the data. The result is a degenerate answer key: most statements evaluate
False on most episodes, so ``FFFF`` dominates and a solver that always answers
``FFFF`` beats one that reads the signal.

This script fits each threshold to the population instead. For every statement
it collects the episode windows the generator would actually sample, sweeps the
threshold, and picks the value whose true-rate is closest to 50%. Statements
are evaluated through ``evaluate_mc_statement`` itself rather than by
reimplementing the statistic, so the calibration cannot drift from the scorer.

It also reports the *undeterminable* rate per statement. Undeterminable is the
second half of the imbalance: ``build_multiselect_options_and_answer`` maps a
``None`` verdict to "F", so a statement about a channel the episode does not
even record is silently scored as a confident False. Thresholds cannot fix
that; those statements need excluding from the option pool for that episode.
See ``--report-only`` output before changing anything.

Usage:
    python scripts/calibrate_l2_tmpl3_thresholds.py --episodes 400
    python scripts/calibrate_l2_tmpl3_thresholds.py --episodes 400 --write
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.question_generation.level2.level2 import (  # noqa: E402
    CONTEXT_MAX,
    CONTEXT_MIN,
    MIN_POST_EVENT_TIMESTAMPS_AFTER,
    _first_timestamp_ms,
    _legacy_mc_option_id,
    normalize_timestamps,
    sample_subseries_before_event,
    split_event_segment,
)
from src.question_generation.utils.io import load_json  # noqa: E402


def load_episode(path: Path) -> Any:
    """Episode rows. Mirrors the generator's nested loader, which unwraps the
    combined {"baseline": [...], "counterfactual": [...]} form."""
    raw = load_json(path)
    if isinstance(raw, dict):
        raw = raw.get("counterfactual") or raw.get("baseline", [])
    return raw


def load_meta(ep_path: Path) -> Dict[str, Any]:
    meta_path = ep_path.with_name(ep_path.stem + "_metadata.json")
    if not meta_path.is_file():
        return {}
    try:
        meta = load_json(meta_path)
    except Exception:
        return {}
    return meta if isinstance(meta, dict) else {}
from src.question_generation.level2.mc_truth import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    evaluate_mc_statement,
)

# Threshold key -> the statement whose truth it gates. Multi-knob statements
# list their magnitude knob first; only that one is calibrated, because the
# second knob (coverage / axis count) controls robustness to noise rather than
# how hard the statement is to satisfy, and moving it changes what the
# statement means rather than how often it fires.
STATEMENT_KEYS: Dict[str, List[str]] = {
    "mc_003": ["speed_drop_ratio"],
    "mc_004": ["speed_stable_tol"],
    "mc_005": ["stall_current_increase"],
    "mc_006": ["force_low_increase"],
    "mc_007": ["force_spike_increase"],
    "mc_008": ["tracking_increase"],
    "mc_009": ["tracking_stable_increase"],
    "mc_010": ["vibration_spike"],
    "mc_011": ["vibration_nominal_band"],
    "mc_012": ["current_peak_increase"],
    "mc_013": ["robot_current_stable_range"],
    "mc_014": ["robot_current_increase"],
    "mc_015": ["tcp_tracking_stable_increase"],
    "mc_016": ["tcp_tracking_increase"],
    "mc_017": ["temp_rise_slope"],
    "mc_018": ["temp_stable_slope"],
    "mc_019": ["no_effect_agg_increase"],
}

# Sweep grid per key, as multiples of the current default. Wide enough that a
# badly-centred default can still be pulled to the median.
SWEEP_MULTIPLIERS = [
    0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0,
    1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0,
]

TARGET_TRUE_RATE = 0.50


def collect_windows(episodes_dir: Path, n_episodes: int, seed: int) -> List[Tuple[list, list, dict]]:
    """Sample (subseries, post_event_rows, metadata) exactly as the generator does."""
    rng = random.Random(seed)
    paths = sorted(p for p in episodes_dir.rglob("*.json") if not p.stem.endswith("_metadata"))
    rng.shuffle(paths)

    windows = []
    for path in paths:
        if len(windows) >= n_episodes:
            break
        try:
            rows = load_episode(path)
        except Exception:
            continue
        if not isinstance(rows, list) or len(rows) < CONTEXT_MIN:
            continue
        meta = load_meta(path)
        # Template 3 is an event template: it needs a window sitting before an
        # event with enough post-event rows to judge the consequence.
        if meta.get("fault_id") in (None, 0, 0.0):
            continue
        try:
            sampled = sample_subseries_before_event(
                rows, CONTEXT_MIN, CONTEXT_MAX,
                min_post_event_after=MIN_POST_EVENT_TIMESTAMPS_AFTER,
                return_metadata=True,
            )
        except Exception:
            continue
        subseries, post_event_rows = sampled[0], sampled[1]
        if not subseries or not post_event_rows:
            continue
        base = _first_timestamp_ms(subseries)
        subseries = normalize_timestamps(subseries, base)
        post_event_rows = normalize_timestamps(post_event_rows, base)
        event_rows, _ = split_event_segment(post_event_rows)
        if not event_rows:
            continue
        windows.append((subseries, post_event_rows, meta))
    return windows


def rate_at(mc_id: str, key: str, value: float, windows, base: Dict[str, float]) -> Tuple[float, float, int]:
    """Return (true_rate_over_determinable, undeterminable_rate, n_determinable)."""
    thresholds = dict(base)
    thresholds[key] = value
    true_n = det_n = none_n = 0
    for subseries, post_event_rows, meta in windows:
        try:
            verdict = evaluate_mc_statement(
                _legacy_mc_option_id(mc_id),
                subseries=subseries,
                post_event_rows=post_event_rows,
                thresholds=thresholds,
                episode_metadata=meta,
            )
        except Exception:
            verdict = None
        if verdict is None:
            none_n += 1
            continue
        det_n += 1
        true_n += bool(verdict)
    total = det_n + none_n
    return (true_n / det_n if det_n else float("nan"),
            none_n / total if total else float("nan"),
            det_n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes-dir", type=Path, default=Path("data/normalized_episodes"))
    ap.add_argument("--episodes", type=int, default=400, help="How many event windows to sample.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write", action="store_true",
                    help="Write the calibrated centres to the JSON file below. Without it, report only.")
    ap.add_argument("--out", type=Path, default=Path("data/mc_options/l2_tmpl3_calibrated_thresholds.json"))
    args = ap.parse_args()

    print(f"sampling up to {args.episodes} event windows from {args.episodes_dir} ...")
    windows = collect_windows(args.episodes_dir, args.episodes, args.seed)
    print(f"collected {len(windows)} windows\n")
    if not windows:
        print("No usable windows. Check --episodes-dir.")
        return 1

    print(f"{'stmt':8s} {'key':30s} {'default':>9s} {'true@def':>9s} "
          f"{'undet':>7s} {'calibrated':>11s} {'true@cal':>9s}")
    calibrated: Dict[str, float] = {}
    unfixable: List[str] = []

    for mc_id, keys in STATEMENT_KEYS.items():
        key = keys[0]
        base = dict(DEFAULT_THRESHOLDS)
        default = float(DEFAULT_THRESHOLDS[key])
        cur_rate, undet, n_det = rate_at(mc_id, key, default, windows, base)

        if n_det == 0:
            print(f"{mc_id:8s} {key:30s} {default:9.4f} {'n/a':>9s} "
                  f"{undet:6.0%} {'-':>11s} {'-':>9s}   NEVER DETERMINABLE")
            unfixable.append(mc_id)
            continue

        best_val, best_rate, best_gap = default, cur_rate, abs(cur_rate - TARGET_TRUE_RATE)
        for mult in SWEEP_MULTIPLIERS:
            val = default * mult
            rate, _, det = rate_at(mc_id, key, val, windows, base)
            if det == 0 or rate != rate:
                continue
            gap = abs(rate - TARGET_TRUE_RATE)
            if gap < best_gap:
                best_val, best_rate, best_gap = val, rate, gap

        calibrated[key] = round(best_val, 6)
        flag = "" if best_gap <= 0.10 else "   <- cannot reach 50%"
        print(f"{mc_id:8s} {key:30s} {default:9.4f} {cur_rate:8.0%} "
              f"{undet:6.0%} {best_val:11.4f} {best_rate:8.0%}{flag}")

    print()
    rates = [rate_at(m, STATEMENT_KEYS[m][0], calibrated.get(STATEMENT_KEYS[m][0], DEFAULT_THRESHOLDS[STATEMENT_KEYS[m][0]]), windows, dict(DEFAULT_THRESHOLDS))[0]
             for m in STATEMENT_KEYS if m not in unfixable]
    rates = [r for r in rates if r == r]
    if rates:
        print(f"after calibration: mean true-rate {statistics.mean(rates):.1%}, "
              f"min {min(rates):.0%}, max {max(rates):.0%}")
    if unfixable:
        print(f"\nnever determinable on this population (always scored F today): {', '.join(unfixable)}")
        print("These need excluding from the option pool, not re-thresholding.")

    if args.write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(calibrated, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    else:
        print("\nreport only. Re-run with --write to persist the calibrated centres.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
