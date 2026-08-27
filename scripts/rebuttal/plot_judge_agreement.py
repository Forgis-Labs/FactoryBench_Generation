#!/usr/bin/env python
"""Paper-quality figures for the L4 inter-judge reliability analysis.

Reads output/judge_agreement/results.json (from judge_agreement.py) and emits:
  fig_judge_agreement.pdf  -- leniency, pairwise agreement/kappa, consensus
  fig_judge_confusion.pdf  -- 3 pairwise 3x3 confusion heatmaps

Forgis palette, embedded fonts, NeurIPS width -- consistent with the probing figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ORANGE = "#FF5A00"; FIRE = "#FF4D00"; FLICKER = "#DC4B07"
INK = "#122128"; STEEL = "#878F92"; BLUE = "#3A7BD5"; GREEN = "#2ECC71"
AMBER = "#FFF3CD"
SCORE_COLORS = {"0.0": STEEL, "0.5": "#F2A65A", "1.0": ORANGE}   # strict -> lenient
JUDGES = ["GPT-5.1", "Claude 4.6", "DeepSeek"]


def rc():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.edgecolor": INK, "axes.linewidth": 0.7,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200,
    })


def fig_agreement(R, out):
    o = R["overall"]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.75),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.26, wspace=0.42)

    # (a) pairwise exact agreement (%) + quadratic-weighted kappa
    ax = axes[0]
    pairs = list(o["pairwise"].keys())
    short = ["GPT--Claude", "GPT--DeepSeek", "Claude--DeepSeek"]
    exact = [o["pairwise"][p]["exact_agreement"] for p in pairs]
    wk = [o["pairwise"][p]["weighted_kappa"] for p in pairs]
    x = np.arange(len(pairs)); w = 0.36
    ax.bar(x - w / 2, exact, w, color=ORANGE, edgecolor="white", linewidth=0.6,
           label="exact agreement", zorder=3)
    ax.bar(x + w / 2, wk, w, color=INK, edgecolor="white", linewidth=0.6,
           label="quadratic-weighted $\\kappa$", zorder=3)
    for xi, (e, k) in enumerate(zip(exact, wk)):
        ax.text(xi - w / 2, e + 0.015, f"{e*100:.1f}%", ha="center", va="bottom",
                fontsize=6.2, color=FLICKER)
        ax.text(xi + w / 2, k + 0.015, f"{k:.2f}", ha="center", va="bottom",
                fontsize=6.2, color=INK)
    # 3-way reliability: dashed reference lines (over the bars) + a box in the
    # empty lower area (all bars are >=0.74, so y<0.7 is free).
    fk, ka = o["fleiss_kappa"], o["krippendorff_alpha"]
    ax.axhline(ka, ls=(0, (4, 2)), lw=0.9, color=BLUE, zorder=4)
    ax.axhline(fk, ls=(0, (1, 1.5)), lw=0.9, color=STEEL, zorder=4)
    ax.text(1.0, 0.30,
            f"3-way reliability\nKrippendorff $\\alpha={ka:.2f}$\nFleiss $\\kappa={fk:.2f}$",
            ha="center", va="center", fontsize=6.3, color=INK, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=STEEL, lw=0.5))
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=6.2, rotation=10)
    ax.set_ylim(0, 1.0); ax.set_ylabel("agreement")
    ax.set_title("(a) Pairwise agreement", fontsize=8.2)
    ax.legend(fontsize=6.0, loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=2, handlelength=1.0, columnspacing=1.2, frameon=False)

    # (b) consensus pie with a clean legend beneath
    ax = axes[1]
    cons = o["consensus"]
    vals = [cons["unanimous"], cons["majority"], cons["split"]]
    colors = [ORANGE, "#EBA55A", STEEL]
    wedges, _ = ax.pie(vals, colors=colors, startangle=90, counterclock=False,
                       radius=1.0, wedgeprops=dict(edgecolor="white", linewidth=1.4))
    ax.set(aspect="equal")
    ax.set_ylim(-1.15, 1.15)
    labels = [f"all three agree ({cons['unanimous']*100:.1f}%)",
              f"2-of-3 majority ({cons['majority']*100:.1f}%)",
              f"three-way split ({cons['split']*100:.1f}%)"]
    ax.legend(wedges, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              title=f"$n = {o['n_complete']:,}$".replace(",", "{,}"),
              title_fontsize=6.0, fontsize=6.0, frameon=False,
              handlelength=0.9, handleheight=0.9, labelspacing=0.45, borderpad=0.2)
    ax.set_title("(b) 3-judge consensus", fontsize=8.2)

    fig.savefig(out / "fig_judge_agreement.pdf")
    plt.close(fig)


def fig_confusion(R, out):
    o = R["overall"]
    pairs = list(o["pairwise"].keys())
    cmap = LinearSegmentedColormap.from_list("forgis", ["#FFFFFF", "#FFD9C2", ORANGE])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.80, bottom=0.20, wspace=0.5)
    ticks = ["0", "0.5", "1"]
    for k, p in enumerate(pairs):
        ax = axes[k]
        cm = np.array(o["pairwise"][p]["confusion"], dtype=float)
        rown = cm / cm.sum()                         # fraction of all items
        im = ax.imshow(rown, cmap=cmap, vmin=0, vmax=rown.max())
        for i in range(3):
            for j in range(3):
                frac = rown[i, j]
                ax.text(j, i, f"{frac*100:.0f}%",
                        ha="center", va="center", fontsize=6.6,
                        color="white" if frac > rown.max() * 0.6 else INK)
        a, b = o["pairwise"][p]["a"], o["pairwise"][p]["b"]
        ax.set_xticks(range(3)); ax.set_xticklabels(ticks, fontsize=7)
        ax.set_yticks(range(3)); ax.set_yticklabels(ticks, fontsize=7)
        ax.set_xlabel(b, fontsize=7.5); ax.set_ylabel(a, fontsize=7.5)
        ax.set_title(f"{a} vs {b}\n(exact {o['pairwise'][p]['exact_agreement']*100:.0f}%)",
                     fontsize=7.2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.savefig(out / "fig_judge_confusion.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("output/judge_agreement/results.json"))
    ap.add_argument("--out", type=Path, default=Path("output/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    R = json.load(open(args.results))
    rc()
    fig_agreement(R, args.out)
    fig_confusion(R, args.out)
    print("wrote fig_judge_agreement.pdf, fig_judge_confusion.pdf to", args.out)


if __name__ == "__main__":
    main()
