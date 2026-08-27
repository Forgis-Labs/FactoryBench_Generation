"""forecast — Chronos-Bolt point/quantile forecaster.

Wraps ``amazon/chronos-bolt-small`` (~200 M params, CPU-fast) so the agent
can delegate any "predict future value" subtask. Model is loaded lazily
and cached at module level; the first call downloads the checkpoint.

The tool operates only on channels that live in the item's time series
(the agent doesn't see the whole episode, only the window in the prompt).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

_PIPELINE = None
_PIPELINE_ERR: Optional[str] = None


def _load_pipeline():
    global _PIPELINE, _PIPELINE_ERR
    if _PIPELINE is not None or _PIPELINE_ERR is not None:
        return _PIPELINE
    try:
        # transformers probes for TensorFlow at import time, and a TF build with
        # protobuf-4-incompatible generated code takes the whole import down with
        # "Descriptors cannot be created directly". Chronos only needs the torch
        # backend, so declining the TF probe is enough; without this the tool
        # returns "chronos load failed" for every call and the agent spends its
        # entire budget on a dead tool.
        import os as _os
        _os.environ.setdefault("USE_TF", "0")
        _os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        import torch
        from chronos import ChronosBoltPipeline
        # device_map="cpu" can leave weights on the "meta" device with newer
        # HF transformers builds; load explicitly on CPU and force real
        # materialisation via to_empty()+load_state_dict style… simplest:
        # let HF materialise by using torch_dtype=torch.float32 and no
        # device_map, then move.
        _PIPELINE = ChronosBoltPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            torch_dtype=torch.float32,
        )
        # Move to CPU explicitly to guarantee no meta tensors survive.
        try:
            _PIPELINE.model = _PIPELINE.model.to("cpu")
        except Exception:
            pass
    except Exception as exc:
        _PIPELINE_ERR = f"chronos load failed: {type(exc).__name__}: {exc}"
    return _PIPELINE


class ForecastTool:
    NAME = "forecast"

    def __init__(self, ts: Dict[str, np.ndarray]):
        self.ts = {k: np.asarray(v, dtype=float) for k, v in ts.items()}

    def spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.NAME,
                "description": (
                    "Forecast the value of a channel N steps ahead using Chronos-Bolt "
                    "(pretrained 200M-parameter time-series foundation model). "
                    "Useful for questions of the form 'expected value of <signal> at "
                    "T+N ms/steps', but it is a small general-purpose model and is "
                    "not reliably better than your own reading of the trend. The "
                    "response contains `predicted_value_at_horizon`, its `q10`/`q90` "
                    "band, a `linear_trend_estimate` from the same channel, and a "
                    "`reliability` verdict. Read `reliability` first and only use the "
                    "point estimate when it says CONSISTENT."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel": {"type": "string", "description": "channel name (e.g. 'feedback_pos_0', 'setpoint_pos_5', 'effort_target_torque_2') matching the signal named in the question"},
                        "horizon": {"type": "integer", "description": "number of steps ahead (>=1). For 'T+N ms' questions on this benchmark, use horizon=N."},
                    },
                    "required": ["channel", "horizon"],
                },
            },
        }

    def __call__(self, channel: str, horizon: int, mode: str = "point") -> Dict[str, Any]:
        if channel not in self.ts:
            avail = sorted(self.ts.keys())
            return {"error": f"channel {channel!r} not in the item's time series",
                    "available_channels": avail[:40]}
        pipe = _load_pipeline()
        if pipe is None:
            return {"error": _PIPELINE_ERR or "chronos pipeline unavailable"}
        v = self.ts[channel]
        v = v[np.isfinite(v)]
        if v.size < 4:
            return {"error": f"channel {channel!r} has <4 finite values; too short to forecast"}
        try:
            import torch
            inputs = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
            quantiles, mean = pipe.predict_quantiles(
                inputs=inputs,
                prediction_length=int(horizon),
                quantile_levels=[0.1, 0.5, 0.9],
            )
            q10  = quantiles[0, :, 0].tolist()
            med  = quantiles[0, :, 1].tolist()   # q50
            q90  = quantiles[0, :, 2].tolist()
            idx = int(horizon) - 1
            point = float(med[idx])
            lo, hi = float(q10[idx]), float(q90[idx])

            # A competing estimate the driver gets for free: least-squares slope
            # over the tail of the observed channel, extrapolated to the same
            # step. This is what "read the trend yourself" amounts to, and on
            # this benchmark it is frequently closer than the forecaster.
            tail = v[-min(len(v), 32):]
            naive = float(tail[-1])
            if tail.size >= 4:
                x = np.arange(tail.size, dtype=float)
                slope, intercept = np.polyfit(x, tail, 1)
                naive = float(slope * (tail.size - 1 + int(horizon)) + intercept)

            # Is the forecaster worth trusting here? Compare its own 10-90 band
            # against the channel's observed spread.
            spread = float(np.nanpercentile(v, 95) - np.nanpercentile(v, 5)) or 1.0
            band_ratio = abs(hi - lo) / abs(spread)
            disagreement = abs(point - naive) / (abs(spread) or 1.0)
            # Horizon relative to the evidence the forecaster was given. Asking a
            # 200M model for step 300 from a 60-row window is extrapolation far
            # past its context, and is where it is least trustworthy.
            reach = float(int(horizon)) / max(1.0, float(v.size))
            if reach > 2.0:
                verdict = (f"OUT OF RANGE: horizon {int(horizon)} is {reach:.0f}x the "
                           f"{v.size}-step window this forecast was conditioned on. The "
                           "estimate is an extrapolation far past its evidence. Use your own "
                           "reading of the trend, or `linear_trend_estimate`, and treat this "
                           "number as a weak prior at best.")
            elif band_ratio > 0.5:
                verdict = ("LOW CONFIDENCE: the 10-90 band spans "
                           f"{band_ratio:.0%} of this channel's own range. Prefer your own "
                           "reading of the trend over this point estimate.")
            elif disagreement > 0.5:
                verdict = ("DISAGREEMENT: this estimate differs from a linear extrapolation "
                           f"of the same channel by {disagreement:.0%} of its range. Decide "
                           "which is better supported before answering; do not copy either blindly.")
            else:
                verdict = ("CONSISTENT: narrow band and agrees with a linear extrapolation "
                           "of the same channel.")

            return {
                "channel": channel,
                "horizon": int(horizon),
                # One estimate among several. Read `reliability` before using it.
                "predicted_value_at_horizon": round(point, 4),
                "q10_at_horizon":             round(lo, 4),
                "q90_at_horizon":             round(hi, 4),
                "linear_trend_estimate":      round(naive, 4),
                "reliability":                verdict,
                # Full trajectory kept for cases where the agent needs a
                # different step; don't average or reduce this list to answer.
                "full_median_series":         [round(x, 4) for x in med],
            }
        except Exception as exc:
            return {"error": f"forecast failed: {type(exc).__name__}: {exc}"}
