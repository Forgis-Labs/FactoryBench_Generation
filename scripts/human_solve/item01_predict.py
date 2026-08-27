#!/usr/bin/env python
"""Item 01: predict when the descent to the bin starts.

The released window ends mid-lift, so the asked-for phase is not in it. The
estimate comes from the other 99 released windows, used purely as an unlabelled
corpus: find every window that contains a hold -> lift -> ... -> descend cycle,
measure lift-onset-to-descent-onset, and add the median to item 01's own lift
onset. Nothing here reads _gold.json or _join.csv.

Usage:  python scripts/human_solve/item01_predict.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stages import PACK, load, segment  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "output" / "human_solve" / "work" / "figs"
TARGET = "level1/01_19d0892f"


def reference_cycles() -> list[dict]:
    """Every hold -> lift -> ... -> descend cycle in the pack, keyed on lift onset."""
    out, seen = [], set()
    for r in csv.DictReader((PACK / "index.csv").open(encoding="utf-8")):
        s = load(PACK / r["dir"])
        segs = segment(s)
        for i, g in enumerate(segs):
            if g["label"] != "lift" or i == 0 or segs[i - 1]["label"] != "hold":
                continue
            nxt = next((h for h in segs[i + 1:] if h["label"] == "descend"), None)
            if nxt is None:
                continue
            key = (g["dz_m"], nxt["t0"] - g["t0"])
            if key in seen:      # the same episode is released in two items
                continue
            seen.add(key)
            out.append({"n": int(r["n"]), "s": s, "lift_t0": g["t0"],
                        "desc_t0": nxt["t0"], "gap": nxt["t0"] - g["t0"]})
            break
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    refs = reference_cycles()
    gaps = np.array([r["gap"] for r in refs], float)
    med = float(np.median(gaps))

    tgt = load(PACK / TARGET)
    lift_t0 = next(g["t0"] for g in segment(tgt) if g["label"] == "lift")
    pred = lift_t0 + med
    lo, hi = lift_t0 + gaps.min(), lift_t0 + gaps.max()

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 6),
                                 gridspec_kw={"height_ratios": [2, 1]})

    # every reference cycle, time-shifted so its lift onset sits at zero
    for r in refs:
        z = (r["s"]["xyz"][:, 2] - r["s"]["xyz"][0, 2]) * 100
        z = z - np.interp(r["lift_t0"], r["s"]["t"], z)
        ax.plot(r["s"]["t"] - r["lift_t0"], z, lw=1.0, alpha=0.55, color="#8899aa")
        ax.plot(r["gap"], np.interp(r["desc_t0"], r["s"]["t"], z), "v",
                ms=7, color="#e45756", zorder=3)

    z1 = (tgt["xyz"][:, 2] - tgt["xyz"][0, 2]) * 100
    z1 = z1 - np.interp(lift_t0, tgt["t"], z1)
    ax.plot(tgt["t"] - lift_t0, z1, lw=2.6, color="#111", label="item 01 (released)")
    ax.axvline(0, color="#4c78a8", lw=1.2, ls="--")
    ax.axvspan(gaps.min(), gaps.max(), color="#e45756", alpha=0.12, lw=0)
    ax.axvline(med, color="#e45756", lw=2.0,
               label=f"predicted descent onset  (+{med:.0f} ms)")
    ax.set_xlabel("time since lift onset [ms]")
    ax.set_ylabel("z relative to lift onset [cm]")
    ax.set_title("item 01: the released window ends mid-lift; the descent to the "
                 "bin is inferred\nfrom 7 matching hold→lift→…→descend cycles "
                 "elsewhere in the pack", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)

    bx.eventplot([gaps], colors="#e45756", lineoffsets=0.5, linelengths=0.7)
    bx.axvline(med, color="#e45756", lw=2)
    for r in refs:
        bx.annotate(f"#{r['n']}", (r["gap"], 0.92), fontsize=7, ha="center")
    bx.set_xlim(ax.get_xlim())
    bx.set_yticks([])
    bx.set_xlabel(f"lift onset → descent onset [ms]   "
                  f"n={len(gaps)}  median={med:.0f}  sd={gaps.std(ddof=1):.0f}")
    bx.grid(alpha=0.25, lw=0.5, axis="x")

    fig.tight_layout()
    p = OUT / "item01_prediction.png"
    fig.savefig(p, dpi=160)

    print(f"wrote {p}\n")
    print(f"item 01 lift onset        : {lift_t0:.0f} ms (step 26)")
    print(f"reference gaps (n={len(gaps)})      : {sorted(int(g) for g in gaps)}")
    print(f"median gap                : {med:.0f} ms")
    print(f"ANSWER (descent onset)    : {pred:.0f} ms")
    print(f"plausible band            : [{lo:.0f}, {hi:.0f}] ms")


if __name__ == "__main__":
    main()
