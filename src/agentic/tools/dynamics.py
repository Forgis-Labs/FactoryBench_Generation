"""describe_dynamics — structural summary of the item's window.

The other three analysis tools answer questions the agent already knows how to
ask: a statistic, a forecast, a snippet of Python. None of them address the
thing Levels 1 to 3 actually test, which is reading dense multivariate
telemetry: where the regime changes, which channels move together, and where
the measured response stops tracking its setpoint. Those are visible in the
numbers and invisible in a mean and a standard deviation, so the agent ends up
eyeballing hundreds of rows in the prompt and mis-reading them.

This returns that structure directly, computed from the window the agent was
given and nothing else:

  * change points, from the largest jumps in a smoothed first difference,
    which is where a phase boundary or a fault onset shows up;
  * per-channel drift, the shift in level between the first and last thirds;
  * setpoint tracking error, for any channel whose name pairs a commanded
    signal with its measured counterpart;
  * the strongest cross-channel correlations, which is how a disturbance that
    propagates across joints becomes legible.

No ground truth is touched: the tool sees the same window the model reads.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

# Commanded/measured pairs, by the acronym stems the prompts use and by the
# expanded names. Tracking error between the two is the operational definition
# of a fault in the paper's causal schema.
_PAIRS = (
    ("setpoint_pos", "feedback_pos"), ("sp", "fp"),
    ("setpoint_speed", "feedback_speed"), ("ss", "fs"),
    ("effort_target_torque", "effort_current"), ("ett", "ec"),
)


def _smooth(x: np.ndarray, w: int = 5) -> np.ndarray:
    if x.size < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


class DynamicsTool:
    NAME = "describe_dynamics"

    def __init__(self, ts: Dict[str, np.ndarray]):
        self.ts = {k: np.asarray(v, dtype=float) for k, v in ts.items()}

    def spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.NAME,
                "description": (
                    "Structural summary of this item's time-series window: regime change "
                    "points, per-channel drift, setpoint-vs-feedback tracking error, and the "
                    "strongest cross-channel correlations. Use this when the question asks "
                    "what happened, when it happened, which channel is anomalous, or how the "
                    "machine's behaviour differs from normal. It reads the same window you "
                    "see, but computes the structure exactly rather than by eye."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "optional subset of channels; omit for all",
                        }
                    },
                },
            },
        }

    def __call__(self, channels: List[str] | None = None) -> Dict[str, Any]:
        ts = {k: v for k, v in self.ts.items() if (not channels or k in channels)}
        ts = {k: v for k, v in ts.items() if v.size >= 6 and np.isfinite(v).sum() >= 6}
        if not ts:
            return {"error": "no usable channels", "available_channels": sorted(self.ts)[:40]}

        n = max(v.size for v in ts.values())
        out: Dict[str, Any] = {"n_steps": int(n)}

        # regime changes: biggest smoothed jumps, pooled across channels
        events = []
        for name, v in ts.items():
            v = np.nan_to_num(v, nan=float(np.nanmean(v)) if np.isfinite(v).any() else 0.0)
            d = np.abs(np.diff(_smooth(v)))
            if d.size == 0:
                continue
            scale = float(np.percentile(d, 75)) or float(d.mean()) or 1.0
            idx = int(np.argmax(d))
            if d[idx] > 4.0 * scale:
                events.append({"channel": name, "step": idx + 1,
                               "magnitude_vs_typical": round(float(d[idx] / scale), 1)})
        events.sort(key=lambda e: -e["magnitude_vs_typical"])
        out["change_points"] = events[:6] or "none stand out above the channel's own noise"

        # drift: level shift between the first and last third
        drift = {}
        for name, v in ts.items():
            t = max(2, v.size // 3)
            a, b = np.nanmean(v[:t]), np.nanmean(v[-t:])
            rng = float(np.nanpercentile(v, 95) - np.nanpercentile(v, 5)) or 1.0
            if abs(b - a) > 0.25 * abs(rng):
                drift[name] = {"start_mean": round(float(a), 4), "end_mean": round(float(b), 4),
                               "shift_vs_range": round(float((b - a) / rng), 2)}
        out["drifting_channels"] = drift or "no channel shifts by more than a quarter of its range"

        # setpoint tracking
        track = {}
        for cmd_stem, meas_stem in _PAIRS:
            for name in ts:
                if not name.startswith(cmd_stem):
                    continue
                suffix = name[len(cmd_stem):]
                other = meas_stem + suffix
                if other not in ts:
                    continue
                a, b = ts[name], ts[other]
                m = min(a.size, b.size)
                err = np.abs(a[:m] - b[:m])
                rng = float(np.nanpercentile(a, 95) - np.nanpercentile(a, 5)) or 1.0
                track[f"{name} vs {other}"] = {
                    "mean_abs_error": round(float(np.nanmean(err)), 4),
                    "max_abs_error": round(float(np.nanmax(err)), 4),
                    "worst_step": int(np.nanargmax(err)),
                    "error_vs_setpoint_range": round(float(np.nanmean(err) / rng), 3),
                }
        out["setpoint_tracking"] = track or "no commanded/measured channel pairs in this window"

        # strongest cross-channel correlations
        names = [k for k, v in ts.items() if np.nanstd(v) > 0][:24]
        corrs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = ts[names[i]], ts[names[j]]
                m = min(a.size, b.size)
                if m < 6:
                    continue
                c = np.corrcoef(np.nan_to_num(a[:m]), np.nan_to_num(b[:m]))[0, 1]
                if np.isfinite(c) and abs(c) > 0.85:
                    corrs.append({"pair": [names[i], names[j]], "r": round(float(c), 3)})
        corrs.sort(key=lambda d: -abs(d["r"]))
        out["strong_correlations"] = corrs[:8] or "no channel pair correlates above 0.85"
        return out
