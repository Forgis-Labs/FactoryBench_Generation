#!/usr/bin/env python
"""Stage-reading figures for the human-solve baseline.

Joint angles are a terrible way to see what a pick-and-place arm is doing: a
descent and a retract can look like the same elbow sweep. So everything here is
lifted into task space first (UR forward kinematics), then reduced to the four
traces that actually separate the phases:

  z(t)          height of the flange -- lifts and descents
  r_xy(t)       horizontal travel from the window's first pose -- transfers
  |q_dot|(t)    joint speed norm -- move vs hold
  wrist wobble  small j1..j3 transients while the arm is parked, which is what
                a gripper actuation looks like from the joint encoders

Reads only output/human_solve/pack/. Never touches _gold.json or _join.csv.

Usage:
  python scripts/human_solve/stages.py --item 1        # one detailed panel
  python scripts/human_solve/stages.py --corpus        # all level-1 windows
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "output" / "human_solve" / "pack"
OUT = REPO / "output" / "human_solve" / "work" / "figs"

UR_ALPHA = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]
UR_MODELS = {
    "UR3": ([0, -0.24365, -0.21325, 0, 0, 0],
            [0.1519, 0, 0, 0.11235, 0.08535, 0.0819]),
    "UR5": ([0, -0.425, -0.39225, 0, 0, 0],
            [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]),
    "UR10": ([0, -0.612, -0.5723, 0, 0, 0],
             [0.1273, 0, 0, 0.163941, 0.1157, 0.0922]),
}


def fk_tcp(q_deg: np.ndarray, model: str = "UR5") -> np.ndarray:
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


def legend_of(qpath: Path) -> dict[str, str]:
    txt = qpath.read_text(encoding="utf-8")
    leg = {}
    if "Channel legend:" in txt:
        for ln in txt.split("Channel legend:", 1)[1].strip().splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                leg[k.strip()] = v.strip()
    return leg


def families(cols, legend):
    """{canonical_family: {joint: column}} -- the legend disambiguates fp/fpo."""
    out: dict[str, dict[int, str]] = {}
    for c in cols:
        full = legend.get(c, c)
        m = re.match(r"^(.*?)_?(\d+)$", full)
        fam, idx = (m.group(1).rstrip("_"), int(m.group(2))) if m else (full, 0)
        out.setdefault(fam, {})[idx] = c
    return out


def series_files(d: Path) -> list[Path]:
    """Ranking items ship series_A.csv .. series_D.csv instead of one file."""
    one = d / "series.csv"
    return [one] if one.exists() else sorted(d.glob("series*.csv"))


def load(d: Path, src: Path | None = None) -> dict:
    src = src or series_files(d)[0]
    recs = list(csv.DictReader(src.open(encoding="utf-8")))
    leg = legend_of(d / "question.txt")
    fams = families([c for c in recs[0] if c not in ("step", "t")], leg)

    def grab(fam):
        if fam not in fams:
            return None
        cols = fams[fam]
        return np.array([[float(r.get(cols[j]) or "nan") for j in sorted(cols)]
                         for r in recs], dtype=float)

    t = np.array([float(r["t"]) for r in recs])
    pos = grab("feedback_pos")
    spd = grab("feedback_speed")
    if spd is None and pos is not None:
        dt = np.median(np.diff(t)) / 1000.0
        spd = np.vstack([np.zeros((1, 6)), np.diff(pos, axis=0) / dt])
    xyz = fk_tcp(np.nan_to_num(pos))
    return {"t": t, "pos": pos, "spd": spd, "xyz": xyz, "src": src,
            "dt": float(np.median(np.diff(t)))}


def segment(s: dict, quiet_frac: float = 0.25, k: float = 6.0) -> list[dict]:
    """Split the window into hold/move blocks and label each move in task space.

    The move/hold threshold is a MAD test against the calmest quarter of the
    window, so a long travel cannot inflate its own noise floor. A move is
    called `lift`/`descend` when the vertical share of the displacement
    dominates, `transfer` when the horizontal share does.
    """
    mag = np.linalg.norm(np.nan_to_num(s["spd"]), axis=1)
    quiet = np.sort(mag)[: max(3, int(len(mag) * quiet_frac))]
    base = float(np.median(quiet))
    mad = float(np.median(np.abs(quiet - base)))
    thr = max(base + k * max(mad, 1e-9), 0.04 * float(mag.max() or 1.0))
    moving = mag > thr

    segs, start = [], 0
    for i in range(1, len(moving) + 1):
        if i == len(moving) or moving[i] != moving[start]:
            segs.append({"kind": "move" if moving[start] else "hold",
                         "i0": start, "i1": i})
            start = i
    # drop 1-sample blips back into their neighbour
    segs = [g for g in segs if g["i1"] - g["i0"] > 1] or segs

    xyz = s["xyz"]
    for g in segs:
        a, b = g["i0"], min(g["i1"], len(xyz) - 1)
        dz = float(xyz[b, 2] - xyz[a, 2])
        dxy = float(np.linalg.norm(xyz[b, :2] - xyz[a, :2]))
        g.update(t0=float(s["t"][a]), t1=float(s["t"][b]),
                 n=g["i1"] - g["i0"], dz_m=round(dz, 4), dxy_m=round(dxy, 4))
        if g["kind"] == "hold":
            g["label"] = "hold"
        elif abs(dz) >= dxy:
            g["label"] = "lift" if dz > 0 else "descend"
        else:
            g["label"] = "transfer"
    return segs


COLORS = {"hold": "#d9d9d9", "lift": "#4c78a8",
          "descend": "#e45756", "transfer": "#f2b100"}


def panel(n: int, d: Path, ax_set, title: str):
    s = load(d)
    t, xyz, spd = s["t"], s["xyz"], s["spd"]
    mag = np.linalg.norm(np.nan_to_num(spd), axis=1)
    r_xy = np.linalg.norm(xyz[:, :2] - xyz[0, :2], axis=1)
    segs = segment(s)

    az, ar, av = ax_set
    for g in segs:
        for ax in ax_set:
            ax.axvspan(g["t0"], g["t1"], color=COLORS[g["label"]], alpha=0.35, lw=0)

    az.plot(t, (xyz[:, 2] - xyz[0, 2]) * 100, color="#333", lw=1.6)
    az.set_ylabel("z - z0  [cm]")
    az.set_title(title, fontsize=10, loc="left")
    ar.plot(t, r_xy * 100, color="#333", lw=1.6)
    ar.set_ylabel("horiz. travel [cm]")
    av.plot(t, mag, color="#333", lw=1.6)
    av.set_ylabel("|q̇| [deg/s]")
    av.set_xlabel("t [ms]")
    for ax in ax_set:
        ax.grid(alpha=0.25, lw=0.5)
    return s, segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", type=int, default=1)
    ap.add_argument("--corpus", action="store_true",
                    help="one z-trace row per level-1 window instead")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    index = list(csv.DictReader((PACK / "index.csv").open(encoding="utf-8")))

    if args.corpus:
        rows = [r for r in index if r["level"] == "1"]
        fig, axes = plt.subplots(len(rows), 1, figsize=(9, 1.05 * len(rows)),
                                 sharex=False)
        for ax, r in zip(axes, rows):
            s = load(PACK / r["dir"])
            segs = segment(s)
            z = (s["xyz"][:, 2] - s["xyz"][0, 2]) * 100
            for g in segs:
                ax.axvspan(g["t0"], g["t1"], color=COLORS[g["label"]],
                           alpha=0.4, lw=0)
            ax.plot(s["t"], z, color="#222", lw=1.2)
            ax.set_ylabel(f"#{r['n']}", rotation=0, ha="right", va="center",
                          fontsize=8)
            ax.set_yticks([])
            ax.tick_params(labelsize=7)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5)
                   for c in COLORS.values()]
        axes[0].legend(handles, list(COLORS), ncol=4, fontsize=8,
                       loc="lower left", bbox_to_anchor=(0, 1.05), frameon=False)
        axes[-1].set_xlabel("t [ms]   (z-height per level-1 window, cm)")
        fig.tight_layout()
        p = OUT / "corpus_level1.png"
        fig.savefig(p, dpi=160)
        print(f"wrote {p}")

        # phase-timing table: what a stage of each kind costs in this task
        stats: dict[str, list[float]] = {}
        for r in index:
            s = load(PACK / r["dir"])
            for g in segment(s):
                if g["label"] != "hold":
                    stats.setdefault(g["label"], []).append(g["t1"] - g["t0"])
        print("\nmove-segment durations across all 100 windows (ms):")
        for k, v in sorted(stats.items()):
            v = np.array(v)
            print(f"  {k:<9} n={len(v):>3}  median={np.median(v):>7.0f}  "
                  f"p25={np.percentile(v,25):>7.0f}  p75={np.percentile(v,75):>7.0f}")
        return

    r = next(x for x in index if int(x["n"]) == args.item)
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
    s, segs = panel(args.item, PACK / r["dir"], axes,
                    f"item {args.item}  (level {r['level']}, "
                    f"template {r['template_id']}, {r['n_timesteps']} steps, "
                    f"dt≈{load(PACK / r['dir'])['dt']:.0f} ms)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5)
               for c in COLORS.values()]
    axes[0].legend(handles, list(COLORS), ncol=4, fontsize=8,
                   loc="upper left", frameon=False)
    fig.tight_layout()
    p = OUT / f"item{args.item:02d}_stages.png"
    fig.savefig(p, dpi=160)
    print(f"wrote {p}\n")
    print(json.dumps(segs, indent=2))


if __name__ == "__main__":
    main()
