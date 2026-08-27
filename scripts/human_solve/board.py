#!/usr/bin/env python
"""Per-item feature board: everything needed to call a phase boundary by eye.

One tall figure per item. Every panel shares an x-axis in milliseconds, with the
step index on the top axis, so a boundary you spot in any panel can be read off
directly in the units the answer wants.

Panels, top to bottom (only the ones the item's channels support):

  z            flange height. Lifts and descents live here and nowhere else.
  x / y        horizontal position, split, so a transfer shows its direction.
  travel       cumulative path length and horizontal distance from the start
               pose. Flat = parked, sloped = moving.
  v_tcp        task-space speed and its vertical component. The sign of vz is
               the single most useful trace for descent-vs-lift.
  q            the six joint angles, mean-removed so they share one axis.
  q_dot        the six joint speeds. Which joints lead a move tells you what
               kind of move it is.
  track        |feedback - setpoint| per joint. Spikes here are contact,
               gripper actuation, or a fault, not commanded motion.
  torque       effort_target_torque norm, when present.
  force        est_contact_force norm, when present. Grasp and release show up
               as steps in this trace.

Candidate boundaries are drawn as dashed verticals with their timestamp
printed: task-space changepoints (blue) and hold/move edges (red). They are
suggestions to check, not answers.

Reads only output/human_solve/pack/. Never _gold.json or _join.csv.

Usage:
  python scripts/human_solve/board.py --item 1
  python scripts/human_solve/board.py --all          # every item in the pack
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stages import PACK, load, segment, families, legend_of, series_files  # noqa: E402
from extract import binseg  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "output" / "human_solve" / "work" / "figs"
J = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def extras(d: Path, src: Path) -> dict:
    """Pull the non-kinematic families (setpoint, torque, contact force)."""
    recs = list(csv.DictReader(src.open(encoding="utf-8")))
    fams = families([c for c in recs[0] if c not in ("step", "t")],
                    legend_of(d / "question.txt"))

    def grab(fam):
        if fam not in fams:
            return None
        cols = fams[fam]
        return np.array([[float(r.get(cols[j]) or "nan") for j in sorted(cols)]
                         for r in recs], dtype=float)

    return {"setpoint_pos": grab("setpoint_pos"),
            "effort_target_torque": grab("effort_target_torque"),
            "est_contact_force": grab("est_contact_force"),
            "setpoint_speed": grab("setpoint_speed")}


def board(item: int, index: list[dict], src: Path) -> Path:
    r = next(x for x in index if int(x["n"]) == item)
    d = PACK / r["dir"]
    s = load(d, src)
    ex = extras(d, src)
    t, xyz, pos, spd = s["t"], s["xyz"], s["pos"], s["spd"]
    dt_s = s["dt"] / 1000.0
    n = len(t)

    v = np.vstack([np.zeros((1, 3)), np.diff(xyz, axis=0) / dt_s])   # m/s
    segs = segment(s)

    rows = [("z", 1.6), ("xy", 1.3), ("travel", 1.2), ("v_tcp", 1.4),
            ("q", 1.4), ("q_dot", 1.4)]
    if ex["setpoint_pos"] is not None:
        rows.append(("track", 1.2))
    if ex["effort_target_torque"] is not None:
        rows.append(("torque", 1.1))
    if ex["est_contact_force"] is not None:
        rows.append(("force", 1.1))

    fig, axes = plt.subplots(len(rows), 1, sharex=True,
                             figsize=(11, sum(h for _, h in rows)),
                             gridspec_kw={"height_ratios": [h for _, h in rows]})
    axm = dict(zip([k for k, _ in rows], axes))

    # ---- shading: hold vs move, plus candidate boundary verticals
    colors = {"hold": "#d9d9d9", "lift": "#4c78a8",
              "descend": "#e45756", "transfer": "#f2b100"}
    edges = sorted({g["t0"] for g in segs} | {segs[-1]["t1"]})
    cps = [t[min(c["index"], n - 1)] for c in binseg(np.nan_to_num(xyz), n_bkps=4)
           if c["gain_frac"] > 0.02]

    for ax in axes:
        for g in segs:
            ax.axvspan(g["t0"], g["t1"], color=colors[g["label"]], alpha=0.30, lw=0)
        for e in edges:
            ax.axvline(e, color="#c0392b", ls="--", lw=0.9, alpha=0.8)
        for c in cps:
            ax.axvline(c, color="#2c6fbb", ls=":", lw=1.1, alpha=0.9)
        ax.grid(alpha=0.22, lw=0.5)
        ax.tick_params(labelsize=8)

    # ---- panels
    z = (xyz[:, 2] - xyz[0, 2]) * 100
    axm["z"].plot(t, z, color="#111", lw=2.0)
    axm["z"].set_ylabel("z - z₀\n[cm]", fontsize=8)

    axm["xy"].plot(t, (xyz[:, 0] - xyz[0, 0]) * 100, lw=1.5, color="#1f77b4", label="x")
    axm["xy"].plot(t, (xyz[:, 1] - xyz[0, 1]) * 100, lw=1.5, color="#2ca02c", label="y")
    axm["xy"].set_ylabel("x, y - start\n[cm]", fontsize=8)
    axm["xy"].legend(fontsize=7, ncol=2, frameon=False, loc="upper left")

    path = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    axm["travel"].plot(t, path * 100, lw=1.6, color="#111", label="path length")
    axm["travel"].plot(t, np.linalg.norm(xyz[:, :2] - xyz[0, :2], axis=1) * 100,
                       lw=1.4, color="#f2b100", label="horiz. from start")
    axm["travel"].set_ylabel("travel\n[cm]", fontsize=8)
    axm["travel"].legend(fontsize=7, ncol=2, frameon=False, loc="upper left")

    axm["v_tcp"].plot(t, np.linalg.norm(v, axis=1) * 1000, lw=1.6, color="#111",
                      label="|v|")
    axm["v_tcp"].plot(t, v[:, 2] * 1000, lw=1.6, color="#e45756", label="v_z (signed)")
    axm["v_tcp"].axhline(0, color="#666", lw=0.7)
    axm["v_tcp"].set_ylabel("TCP speed\n[mm/s]", fontsize=8)
    axm["v_tcp"].legend(fontsize=7, ncol=2, frameon=False, loc="upper left")

    for j in range(pos.shape[1]):
        axm["q"].plot(t, pos[:, j] - np.nanmean(pos[:, j]), lw=1.2, color=J[j],
                      label=f"j{j}")
        axm["q_dot"].plot(t, spd[:, j], lw=1.2, color=J[j])
    axm["q"].set_ylabel("q - mean\n[deg]", fontsize=8)
    axm["q"].legend(fontsize=7, ncol=6, frameon=False, loc="upper left")
    axm["q_dot"].set_ylabel("q̇\n[deg/s]", fontsize=8)
    axm["q_dot"].axhline(0, color="#666", lw=0.7)

    if "track" in axm:
        err = np.abs(np.nan_to_num(pos - ex["setpoint_pos"]))
        for j in range(err.shape[1]):
            axm["track"].plot(t, err[:, j], lw=1.1, color=J[j])
        axm["track"].set_ylabel("|fb - sp|\n[deg]", fontsize=8)
    if "torque" in axm:
        axm["torque"].plot(t, np.linalg.norm(np.nan_to_num(ex["effort_target_torque"]),
                                             axis=1), lw=1.5, color="#7d3c98")
        axm["torque"].set_ylabel("|torque|", fontsize=8)
    if "force" in axm:
        axm["force"].plot(t, np.linalg.norm(np.nan_to_num(ex["est_contact_force"]),
                                            axis=1), lw=1.5, color="#c0392b")
        axm["force"].set_ylabel("|contact F|", fontsize=8)

    # ---- boundary timestamps, printed on the top panel
    ymin, ymax = axm["z"].get_ylim()
    for e in edges:
        axm["z"].annotate(f"{e:.0f}", (e, ymax), fontsize=7.5, color="#c0392b",
                          rotation=90, va="top", ha="right",
                          xytext=(-1, -2), textcoords="offset points")
    for c in cps:
        axm["z"].annotate(f"{c:.0f}", (c, ymin), fontsize=7.5, color="#2c6fbb",
                          rotation=90, va="bottom", ha="left",
                          xytext=(1, 2), textcoords="offset points")

    axes[-1].set_xlabel("t [ms]     dashed red = hold/move edge     "
                        "dotted blue = task-space changepoint", fontsize=9)

    # step index on the top axis, same scale
    top = axes[0].secondary_xaxis(
        "top", functions=(lambda ms: np.interp(ms, t, np.arange(n)),
                          lambda k: np.interp(k, np.arange(n), t)))
    top.set_xlabel("step index", fontsize=8)
    top.tick_params(labelsize=8)

    seq = " → ".join(g["label"] for g in segs)
    tag = "" if src.stem == "series" else f"  [{src.stem}]"
    axes[0].set_title(
        f"item {item:02d}{tag}   level {r['level']}, template {r['template_id']} "
        f"({r['template_type']})   {n} steps, dt≈{s['dt']:.0f} ms, "
        f"t∈[0, {t[-1]:.0f}] ms\nsegmentation: {seq}",
        fontsize=10, loc="left", pad=22)

    fig.tight_layout()
    suffix = "" if src.stem == "series" else f"_{src.stem.split('_')[-1]}"
    p = OUT / f"board_{item:02d}{suffix}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", type=int, default=1)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    index = list(csv.DictReader((PACK / "index.csv").open(encoding="utf-8")))
    items = [int(r["n"]) for r in index] if args.all else [args.item]
    for i in items:
        d = PACK / next(r for r in index if int(r["n"]) == i)["dir"]
        for src in series_files(d):
            print(f"wrote {board(i, index, src)}")


if __name__ == "__main__":
    main()
