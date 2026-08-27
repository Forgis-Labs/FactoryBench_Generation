#!/usr/bin/env python
"""Stage 0 of the human-solve pipeline: turn the blinded pack into evidence.

Reads only `output/human_solve/pack/` -- never `_gold.json` or `_join.csv`, so
nothing here can see an answer. Everything it emits is derived from the same
question text and time series a human solver is handed.

What it does per item:

  * routes the item to an answer-format family (F1..F7) from the question's
    own instruction sentence, cross-checked against (level, template_id)
  * pulls the ask out of the prompt: window length, forecast horizon, target
    joint, counterfactual event time, named fault
  * canonicalises channels through the prompt's own acronym legend, so
    `fp3` and `fpo3` both land on feedback_pos[3]
  * lifts joint space into task space with UR3/UR5/UR10 forward kinematics.
    Joint angles alone are misleading: item 01 looks like a large motion in
    joint space and is a pure vertical lift in task space.
  * detects phase structure: motion onset, dwell/move segmentation, and
    binary-segmentation changepoints on both the command and feedback streams
  * for ranking items, parses each option's segment dump into rows so the F7
    solver can chain them

Outputs (all under output/human_solve/work/):

  items/<nn>_<id8>.json   full evidence for one item
  features.csv            one flat row per item, for eyeballing the whole set

Usage:
  python scripts/human_solve/extract.py
  python scripts/human_solve/extract.py --level 1 --item 1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "output" / "human_solve" / "pack"
WORK = REPO / "output" / "human_solve" / "work"

# ---------------------------------------------------------------- kinematics

# Standard DH (a, d, alpha) for the UR arms. Alpha is shared across models.
UR_ALPHA = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]
UR_MODELS = {
    "UR3": ([0, -0.24365, -0.21325, 0, 0, 0],
            [0.1519, 0, 0, 0.11235, 0.08535, 0.0819]),
    "UR5": ([0, -0.425, -0.39225, 0, 0, 0],
            [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]),
    "UR10": ([0, -0.612, -0.5723, 0, 0, 0],
             [0.1273, 0, 0, 0.163941, 0.1157, 0.0922]),
}


def fk_tcp(q_deg: np.ndarray, model: str) -> np.ndarray:
    """Forward kinematics for a T x 6 array of joint angles in degrees.

    Returns T x 3 TCP positions in metres. The three UR models differ only in
    link scale, so running all of them and keeping the ones that agree in sign
    guards against reading a lift as a descent on the wrong robot.
    """
    a, d = UR_MODELS[model]
    q = np.radians(q_deg)
    out = np.empty((len(q), 3))
    for k, row in enumerate(q):
        T = np.eye(4)
        for i in range(6):
            ct, st = math.cos(row[i]), math.sin(row[i])
            ca, sa = math.cos(UR_ALPHA[i]), math.sin(UR_ALPHA[i])
            T = T @ np.array([
                [ct, -st * ca,  st * sa, a[i] * ct],
                [st,  ct * ca, -ct * sa, a[i] * st],
                [0,        sa,       ca,      d[i]],
                [0,         0,        0,         1],
            ])
        out[k] = T[:3, 3]
    return out


# ------------------------------------------------------------- changepoints

def _seg_cost(cs: np.ndarray, cs2: np.ndarray, a: int, b: int) -> float:
    """Sum of squared deviations from the mean over rows [a, b), all columns."""
    n = b - a
    if n <= 0:
        return 0.0
    s = cs[b] - cs[a]
    s2 = cs2[b] - cs2[a]
    return float(np.sum(s2 - s * s / n))


def binseg(X: np.ndarray, n_bkps: int = 3, min_size: int = 3) -> list[dict]:
    """Greedy binary segmentation on a T x C matrix.

    Each returned breakpoint carries the variance it explained, normalised by
    the total, so a weak split is visibly weak instead of silently ranked
    first. Columns are z-scored so a large-amplitude channel cannot drown the
    rest.
    """
    T = len(X)
    if T < 2 * min_size:
        return []
    Z = X - X.mean(0)
    sd = Z.std(0)
    Z = Z / np.where(sd > 1e-12, sd, 1.0)
    Z = Z[:, sd > 1e-12] if np.any(sd > 1e-12) else Z
    if Z.size == 0:
        return []
    cs = np.vstack([np.zeros(Z.shape[1]), np.cumsum(Z, 0)])
    cs2 = np.vstack([np.zeros(Z.shape[1]), np.cumsum(Z ** 2, 0)])
    total = _seg_cost(cs, cs2, 0, T) or 1.0

    bounds = [0, T]
    found = []
    for _ in range(n_bkps):
        best = None
        for a, b in zip(bounds[:-1], bounds[1:]):
            base = _seg_cost(cs, cs2, a, b)
            for k in range(a + min_size, b - min_size + 1):
                gain = base - _seg_cost(cs, cs2, a, k) - _seg_cost(cs, cs2, k, b)
                if best is None or gain > best[0]:
                    best = (gain, k)
        if best is None or best[0] <= 0:
            break
        found.append({"index": int(best[1]),
                      "gain_frac": round(best[0] / total, 4)})
        bounds = sorted(bounds + [best[1]])
    return found


def motion_onset(mag: np.ndarray, k: float = 5.0, hold: int = 2) -> int | None:
    """First index where `mag` breaks out of its quiet baseline and stays out.

    Baseline is the median/MAD of the calmest quarter of the window rather than
    the whole window, so a long tail of motion cannot inflate the threshold.
    """
    if len(mag) < 6:
        return None
    quiet = np.sort(mag)[: max(3, len(mag) // 4)]
    base, mad = float(np.median(quiet)), float(np.median(np.abs(quiet - np.median(quiet))))
    thr = base + k * max(mad, 1e-9)
    over = mag > thr
    for i in range(len(over) - hold + 1):
        if over[i:i + hold].all():
            return int(i)
    return None


def dwell_segments(mag: np.ndarray) -> list[dict]:
    """Run-length encode the window into `hold` and `move` blocks."""
    if len(mag) == 0:
        return []
    quiet = np.sort(mag)[: max(3, len(mag) // 4)]
    base = float(np.median(quiet))
    mad = float(np.median(np.abs(quiet - base)))
    thr = max(base + 5 * mad, 0.05 * float(mag.max()) if mag.max() > 0 else 0.0)
    moving = mag > thr
    segs, start = [], 0
    for i in range(1, len(moving) + 1):
        if i == len(moving) or moving[i] != moving[start]:
            segs.append({"kind": "move" if moving[start] else "hold",
                         "start": start, "end": i, "length": i - start})
            start = i
    return segs


# ------------------------------------------------------------------ parsing

OPT_RE = re.compile(r"^ {2}([A-Z])\)\s(.*)$")


def load_question(path: Path) -> dict:
    """Split question.txt into prompt, options and acronym legend."""
    txt = path.read_text(encoding="utf-8")
    legend = {}
    body = txt
    if "Channel legend:" in txt:
        body, leg = txt.split("Channel legend:", 1)
        for ln in leg.strip().splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                legend[k.strip()] = v.strip()

    options: dict[str, str] = {}
    prompt_lines, last = [], None
    in_opts = False
    for ln in body.splitlines():
        if ln.strip() == "Options:":
            in_opts = True
            continue
        if re.match(r"^Series[ _A-Z]*:", ln.strip()):
            in_opts = False
            continue
        if in_opts:
            m = OPT_RE.match(ln)
            if m:
                last = m.group(1)
                options[last] = m.group(2)
            elif ln.strip() and last:
                # an option that wrapped onto the next line
                options[last] += " " + ln.strip()
        elif ln.strip() and not ln.startswith("# item"):
            prompt_lines.append(ln.strip())
    return {"prompt": " ".join(prompt_lines), "options": options, "legend": legend}


def canon_channels(cols: list[str], legend: dict[str, str]) -> dict[str, dict[int, str]]:
    """Map raw column names onto {canonical_family: {joint_index: column}}.

    The legend in the prompt is authoritative: it is what tells us `fpo3` and
    `fp3` are both feedback_pos joint 3. Columns with no legend entry fall back
    to stripping the trailing digits, which keeps unknown channels visible
    instead of dropping them.
    """
    out: dict[str, dict[int, str]] = {}
    for c in cols:
        full = legend.get(c)
        if full:
            m = re.match(r"^(.*?)_(\d+)$", full)
            fam, idx = (m.group(1), int(m.group(2))) if m else (full, 0)
        else:
            m = re.match(r"^([a-z]+)(\d+)$", c)
            fam, idx = (m.group(1), int(m.group(2))) if m else (c, 0)
        out.setdefault(fam, {})[idx] = c
    return out


def stack(recs: list[dict], cols: dict[int, str]) -> np.ndarray:
    """Pull a family's channels into a T x J array, joints in index order."""
    order = sorted(cols)
    return np.array([[float(r.get(cols[j], "nan") or "nan") for j in order]
                     for r in recs], dtype=float)


def parse_option_segments(text: str) -> list[dict]:
    """Parse a ranking option ('k=v, k=v | k=v, k=v') into a list of rows."""
    rows = []
    for chunk in text.split("|"):
        rec = {}
        for kv in chunk.split(","):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            try:
                rec[k.strip()] = float(v)
            except ValueError:
                continue
        if rec:
            rows.append(rec)
    return rows


# ------------------------------------------------------------------ routing

# Ordered: the first cue that matches wins. Cues come from the answer-format
# sentence, which is the thing that actually determines how an item is solved.
CUES = [
    ("F7", "rank the signal segments"),
    ("F1", "at which timestamp should the window begin"),
    ("F3", "what robot does this sensor data originate from"),
    ("F5", "what anomaly is present"),
    ("F4", "4 letter string using f and t"),
    ("F2", "expected value"),
]
# (level, template_id) -> family, used only to flag disagreement with the cue.
TABLE = {
    (1, 1): "F1", (2, 6): "F1",
    (1, 7): "F2", (2, 4): "F2", (2, 5): "F2", (3, 4): "F2", (3, 5): "F2",
    (1, 6): "F3", (2, 10): "F3",
    (1, 3): "F4", (2, 2): "F4", (3, 2): "F4", (3, 3): "F4",
    (2, 7): "F5",
    (4, 1): "F6", (4, 2): "F6",
    (2, 1): "F7", (3, 1): "F7",
}


def route(prompt: str, level: int, tid: int) -> tuple[str, str]:
    low = prompt.lower()
    for fam, cue in CUES:
        if cue in low:
            return fam, "cue"
    if level == 4:
        return "F6", "level"
    return TABLE.get((level, tid), "F?"), "table"


def f1_generator_prior(window_length: int, n_steps: int) -> dict:
    """Bound the F1 answer index from how the generator built the question.

    Level 1 template 1 sets the quoted window to `phase_length + 5`, and the
    phase it picks is an interior segment of the released window, so it can be
    neither the first nor the last block. That alone pins the start index to a
    narrow band before any signal is examined -- which is a leak in the
    benchmark, not a solving technique. Kept behind --generator-prior so a run
    that uses it is distinguishable from one that does not.
    """
    phase_len = window_length - 5
    lo, hi = 1, max(1, n_steps - phase_len - 1)
    return {"implied_phase_length": phase_len,
            "feasible_start_index": [lo, hi],
            "note": "derived from level1.py window_length = phase_length + 5"}


def read_ask(prompt: str) -> dict:
    """Pull the quantitative ask out of the prompt text."""
    def one(pat, cast=int):
        m = re.search(pat, prompt, re.I)
        return cast(m.group(1)) if m else None

    joints = re.findall(r"joint (\d+)", prompt, re.I)
    quantity = None
    for phrase in ("commanded joint velocities", "joint velocities",
                   "commanded position", "velocity of joint",
                   "position of joint"):
        if phrase in prompt.lower():
            quantity = phrase
            break
    fault = None
    for pat in (r"suffers from (?:an?\s+)?(.+?) in the given",
                r"exhibiting (?:an?\s+)?(.+?)[.?]",
                r"where (?:an?\s+)?(.+?) occurs at timestep"):
        m = re.search(pat, prompt, re.I)
        if m and m.group(1).strip():
            fault = m.group(1).strip()
            break
    return {
        "window_length": one(r"fixed window length of (\d+) timesteps"),
        "horizon_ms": one(r"at T\+(\d+)\s*ms"),
        "event_ms": one(r"occurs at timestep (\d+)\s*ms"),
        "target_joint": int(joints[0]) if len(joints) == 1 else None,
        "target_quantity": quantity,
        "named_fault": fault,
    }


# ---------------------------------------------------------------- per-series

def analyse_series(path: Path, legend: dict) -> dict:
    with path.open(encoding="utf-8") as f:
        recs = list(csv.DictReader(f))
    if not recs:
        return {"name": path.stem, "n": 0}
    cols = [c for c in recs[0] if c not in ("step", "t")]
    fams = canon_channels(cols, legend)
    t = np.array([float(r["t"]) for r in recs])
    dt = np.diff(t)

    out: dict = {
        "name": path.stem,
        "n": len(recs),
        "t_range_ms": [float(t[0]), float(t[-1])],
        "dt_ms": {"median": float(np.median(dt)) if len(dt) else None,
                  "min": float(dt.min()) if len(dt) else None,
                  "max": float(dt.max()) if len(dt) else None},
        "channel_families": {k: len(v) for k, v in sorted(fams.items())},
    }

    pos = stack(recs, fams["feedback_pos"]) if "feedback_pos" in fams else None
    cmd = stack(recs, fams["setpoint_pos"]) if "setpoint_pos" in fams else None
    spd = stack(recs, fams["feedback_speed"]) if "feedback_speed" in fams else None

    # velocity magnitude: prefer the reported speed, else differentiate position
    if spd is not None:
        mag = np.linalg.norm(np.nan_to_num(spd), axis=1)
    elif pos is not None:
        dq = np.vstack([np.zeros((1, pos.shape[1])), np.diff(pos, axis=0)])
        mag = np.linalg.norm(np.nan_to_num(dq), axis=1)
    else:
        mag = np.zeros(len(recs))

    if pos is not None and pos.shape[1] == 6:
        out["joint_range_deg"] = {
            f"j{j}": [round(float(np.nanmin(pos[:, j])), 3),
                      round(float(np.nanmax(pos[:, j])), 3)]
            for j in range(6)}
        out["fk"] = {}
        for model in UR_MODELS:
            xyz = fk_tcp(np.nan_to_num(pos), model)
            span = xyz.max(0) - xyz.min(0)
            out["fk"][model] = {
                "xyz_span_m": [round(float(v), 4) for v in span],
                "z_start_m": round(float(xyz[0, 2]), 4),
                "z_end_m": round(float(xyz[-1, 2]), 4),
                "z_net_m": round(float(xyz[-1, 2] - xyz[0, 2]), 4),
                "dominant_axis": ["x", "y", "z"][int(np.argmax(span))],
                "path_len_m": round(float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()), 4),
            }
    if spd is not None:
        out["peak_abs_speed"] = {f"j{j}": round(float(np.nanmax(np.abs(spd[:, j]))), 3)
                                 for j in range(spd.shape[1])}
    if pos is not None and cmd is not None and pos.shape == cmd.shape:
        err = np.nan_to_num(pos - cmd)
        out["tracking_error"] = {
            "rms_per_joint": [round(float(np.sqrt((err[:, j] ** 2).mean())), 4)
                              for j in range(err.shape[1])],
            "max_abs": round(float(np.abs(err).max()), 4),
            "argmax_step": int(np.abs(err).sum(1).argmax()),
        }
    for fam, key in (("est_contact_force", "contact"),
                     ("effort_target_torque", "torque")):
        if fam in fams:
            M = np.nan_to_num(stack(recs, fams[fam]))
            n = np.linalg.norm(M, axis=1)
            out[key] = {"peak": round(float(n.max()), 4),
                        "peak_step": int(n.argmax()),
                        "mean": round(float(n.mean()), 4)}

    segs = dwell_segments(mag)
    moves = [s for s in segs if s["kind"] == "move"]
    # two different questions, kept apart because they disagree constantly:
    # the MAD test fires on any excursion above the noise floor (a gripper
    # actuation shows up there), the segmentation gives the first block of
    # sustained travel. On item 01 that is step 12 versus step 26.
    out["motion"] = {
        "speed_norm_max": round(float(mag.max()), 4),
        "first_excursion_step": motion_onset(mag),
        "sustained_onset_step": moves[0]["start"] if moves else None,
        "segments": segs,
    }
    out["changepoints"] = {
        "feedback": binseg(np.nan_to_num(spd if spd is not None else
                                         (pos if pos is not None else mag[:, None]))),
        "command": binseg(np.vstack([np.zeros((1, cmd.shape[1])), np.diff(cmd, axis=0)]))
        if cmd is not None else [],
    }
    return out


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--item", type=int, default=None, help="single item number")
    ap.add_argument("--generator-prior", action="store_true",
                    help="emit the F1 index bound implied by the generator "
                         "(a benchmark leak, off by default)")
    args = ap.parse_args()

    with (PACK / "index.csv").open(encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    if args.level:
        index = [r for r in index if int(r["level"]) == args.level]
    if args.item:
        index = [r for r in index if int(r["n"]) == args.item]

    (WORK / "items").mkdir(parents=True, exist_ok=True)
    flat, mismatches, fam_counts = [], [], {}

    for r in index:
        d = PACK / r["dir"]
        level, tid = int(r["level"]), int(r["template_id"])
        q = load_question(d / "question.txt")
        family, how = route(q["prompt"], level, tid)
        if TABLE.get((level, tid)) not in (None, family):
            mismatches.append((r["n"], r["dir"], family, TABLE[(level, tid)]))
        fam_counts[family] = fam_counts.get(family, 0) + 1

        series = [analyse_series(p, q["legend"])
                  for p in sorted(d.glob("series*.csv"))]
        item = {
            "n": int(r["n"]), "id": r["id"], "dir": r["dir"],
            "level": level, "template_id": tid,
            "template_type": r["template_type"],
            "family": family, "family_source": how,
            "prompt": q["prompt"],
            "ask": read_ask(q["prompt"]),
            "options": q["options"],
            "series": series,
        }
        if family == "F7":
            item["option_segments"] = {k: parse_option_segments(v)
                                       for k, v in q["options"].items()}
        if args.generator_prior and family == "F1" and item["ask"]["window_length"]:
            item["generator_prior"] = f1_generator_prior(
                item["ask"]["window_length"], series[0].get("n", 0))

        stem = d.name
        (WORK / "items" / f"{stem}.json").write_text(
            json.dumps(item, indent=2), encoding="utf-8")

        s0 = series[0] if series else {}
        fk = (s0.get("fk") or {}).get("UR5", {})
        flat.append({
            "n": r["n"], "level": level, "template_id": tid, "family": family,
            "n_series": len(series), "n_steps": s0.get("n", 0),
            "dt_ms": (s0.get("dt_ms") or {}).get("median"),
            "channels": ";".join(f"{k}:{v}" for k, v in
                                 (s0.get("channel_families") or {}).items()),
            "onset_step": (s0.get("motion") or {}).get("sustained_onset_step"),
            "first_excursion": (s0.get("motion") or {}).get("first_excursion_step"),
            "cp_command": ";".join(str(c["index"]) for c in
                                   (s0.get("changepoints") or {}).get("command", [])),
            "cp_feedback": ";".join(str(c["index"]) for c in
                                    (s0.get("changepoints") or {}).get("feedback", [])),
            "fk_dominant_axis": fk.get("dominant_axis"),
            "fk_z_net_m": fk.get("z_net_m"),
            "track_max_abs": (s0.get("tracking_error") or {}).get("max_abs"),
            "contact_peak": (s0.get("contact") or {}).get("peak"),
            "window_length": item["ask"]["window_length"],
            "horizon_ms": item["ask"]["horizon_ms"],
            "event_ms": item["ask"]["event_ms"],
            "named_fault": item["ask"]["named_fault"],
        })

    with (WORK / "features.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)

    print(f"{len(flat)} items -> {WORK / 'items'}")
    print("family mix: " + ", ".join(f"{k}={v}" for k, v in sorted(fam_counts.items())))
    if mismatches:
        print(f"routing disagreement on {len(mismatches)} item(s):")
        for n, dd, cue, tab in mismatches:
            print(f"   item {n} {dd}: cue={cue} table={tab}")
    else:
        print("routing: cue and (level, template) agree on every item")
    print(f"summary: {WORK / 'features.csv'}")


if __name__ == "__main__":
    main()
