"""Two-panel figure for the paper:

  (a)  The loop the driver actually runs: choose a tool, run it, read the
       result, repeat up to six times. Drawn as an explicit cycle rather than
       a static box chain, because the iteration is the only structure the
       pipeline has and it is what a reader needs to see.
  (b)  Accuracy per level, zero-shot vs the same model given four tools.

Numbers are read from a hardcoded dict populated from the rescore output
(scripts/rescore_signed.py) so the figure regenerates deterministically
even when the reply tree changes.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Forgis brand palette (matches other paper figures) ────────────────
TIGER     = "#FF5A00"
FLICKER   = "#DC4B07"
STEEL     = "#878F92"
HAIRLINE  = "#DCE0E1"
CREAM     = "#FFE9DA"
PANEL_BG  = "#FBFAF8"
GUNMETAL  = "#122128"


def set_style() -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "axes.linewidth":    0.8,
        "axes.edgecolor":    GUNMETAL,
        "axes.labelcolor":   GUNMETAL,
        "xtick.color":       GUNMETAL,
        "ytick.color":       GUNMETAL,
        "text.color":        GUNMETAL,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
    })


# ── Data ──────────────────────────────────────────────────────────────
# From output/agent_rescore.json + judge-scored L4 replies (200 items / level,
# stratified random subset, seed 42). L4 uses the uncorrected LLM-judge rubric
# (see paper §5.4 caveat: L4 magnitudes not comparable to signed L1–L3).
LEVEL_LABELS  = ["L1  State", "L2  Intervention", "L3  Counterfactual", "L4  Decision"]
ZERO_SHOT_GPT = [15.9,  1.3, 18.2, 16.2]
AGENT_GPT     = [38.8, 15.2, 24.9, 33.2]


def panel_a_loop(ax) -> None:
    """The agent as a state machine: three states in a closed cycle.

    The states are arranged on a circle and the transitions follow it, so the
    cycle is the shape the eye picks up first. Entry and exit hang off the
    reasoning state, which is where the driver decides between calling another
    tool and committing to an answer.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.35)
    ax.axis("off")

    def node(cx, cy, w, h, lines, *, fill="white", edge=HAIRLINE, lw=0.9,
             fs=8.6, bold=False, sub_color=STEEL, z=4):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0,rounding_size=0.10",
            linewidth=lw, edgecolor=edge, facecolor=fill, zorder=z))
        if len(lines) == 1:
            ax.text(cx, cy, lines[0], ha="center", va="center", fontsize=fs,
                    fontweight="bold" if bold else "normal", zorder=z + 1)
        else:
            ax.text(cx, cy + 0.10, lines[0], ha="center", va="center", fontsize=fs,
                    fontweight="bold" if bold else "normal", zorder=z + 1)
            ax.text(cx, cy - 0.14, lines[1], ha="center", va="center",
                    fontsize=fs - 1.2, color=sub_color, zorder=z + 1)

    def edge(x1, y1, x2, y2, *, color=GUNMETAL, lw=1.0, curve=0.0, z=2, ls="-"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
            linewidth=lw, color=color, zorder=z, linestyle=ls,
            connectionstyle=f"arc3,rad={curve}", shrinkA=3, shrinkB=3))

    # ── the three states, on a circle ────────────────────────────────
    RX, RY_ = 5.00, 1.92          # cycle centre
    S_REASON = (RX, RY_ + 0.62)   # top
    S_CALL   = (RX + 1.52, RY_ - 0.50)
    S_READ   = (RX - 1.52, RY_ - 0.50)
    SW, SH = 1.86, 0.60

    # cycle transitions first, so the state boxes sit on top of them
    edge(S_REASON[0] + 0.55, S_REASON[1] - 0.22, S_CALL[0] + 0.10, S_CALL[1] + 0.34,
         color=TIGER, lw=1.25, curve=-0.32, z=2)
    edge(S_CALL[0] - 0.80, S_CALL[1] - 0.10, S_READ[0] + 0.80, S_READ[1] - 0.10,
         color=TIGER, lw=1.25, curve=-0.26, z=2)
    edge(S_READ[0] - 0.10, S_READ[1] + 0.34, S_REASON[0] - 0.55, S_REASON[1] - 0.22,
         color=TIGER, lw=1.25, curve=-0.32, z=2)

    node(*S_REASON, SW, SH, ["reason"], fill=CREAM, edge=TIGER, lw=1.4,
         fs=9.4, bold=True, z=5)
    node(*S_CALL, SW, SH, ["call a tool"], z=5)
    node(*S_READ, SW, SH, ["read the result"], z=5)

    ax.text(RX, RY_ - 0.06, "at most" + chr(10) + "six times",
            ha="center", va="center",
            fontsize=7.6, color=FLICKER, style="italic", linespacing=1.35, zorder=6)

    # ── entry and exit, both on the reasoning state ──────────────────
    node(0.92, S_REASON[1], 1.56, 0.74,
         ["Benchmark item", "series + question"], fs=8.2)
    node(9.08, S_REASON[1], 1.56, 0.74, ["Answer", "item's format"], fs=8.2)
    edge(1.70, S_REASON[1], S_REASON[0] - SW / 2, S_REASON[1], lw=1.1, z=3)
    edge(S_REASON[0] + SW / 2, S_REASON[1], 8.30, S_REASON[1], lw=1.1, z=3)
    ax.text((S_REASON[0] + SW / 2 + 8.30) / 2, S_REASON[1] + 0.20,
            "when done", ha="center", va="bottom", fontsize=7.4,
            color=STEEL, style="italic")

    # ── the tool menu, reachable only from the call state ────────────
    TY, TH = 0.20, 0.52
    TX, TW = 1.55, 6.90
    ax.add_patch(FancyBboxPatch(
        (TX, TY), TW, TH, boxstyle="round,pad=0,rounding_size=0.07",
        linewidth=0.9, edgecolor=HAIRLINE, facecolor=PANEL_BG, zorder=2))
    edge(S_CALL[0], S_CALL[1] - SH / 2, S_CALL[0], TY + TH,
         color=STEEL, lw=0.9, ls=(0, (2, 2)), z=3)

    tools = ["signal statistics", "forecaster", "numerical sandbox", "manual retriever"]
    slot = TW / len(tools)
    for i, name in enumerate(tools):
        ax.text(TX + slot * (i + 0.5), TY + TH / 2, name, ha="center", va="center",
                fontsize=7.8, zorder=4)
        if i:
            ax.plot([TX + slot * i, TX + slot * i], [TY + 0.09, TY + TH - 0.09],
                    color=HAIRLINE, lw=0.8, zorder=3)


def plot_lift(ax) -> None:
    """Accuracy per level, zero-shot vs the same model with tools."""
    x = np.arange(len(LEVEL_LABELS))
    w = 0.36

    ax.bar(x - w / 2, ZERO_SHOT_GPT, w, label="GPT-5.1, zero-shot",
           color="white", edgecolor=STEEL, linewidth=1.2, zorder=3)
    ax.bar(x + w / 2, AGENT_GPT, w, label="GPT-5.1 + four tools (ours)",
           color=TIGER, edgecolor=FLICKER, linewidth=0.8, zorder=3)

    for xi, zs, ag in zip(x, ZERO_SHOT_GPT, AGENT_GPT):
        ax.annotate(f"+{ag - zs:.1f}", xy=(xi + w / 2, ag), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=FLICKER, zorder=5)
        ax.annotate(f"{zs:.1f}", xy=(xi - w / 2, zs), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=STEEL, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 46)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.25, color=STEEL)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.spines["left"].set_color(STEEL)
    ax.spines["bottom"].set_color(STEEL)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              frameon=False, fontsize=9, handlelength=1.3,
              columnspacing=1.6, borderpad=0.0, handletextpad=0.5)


def main() -> None:
    set_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.6),
                             gridspec_kw={"height_ratios": [2.2, 3.3]})
    panel_a_loop(axes[0])
    axes[0].set_title("(a)  The agent's state loop", fontsize=10.5,
                      fontweight="bold", loc="left", pad=2)
    plot_lift(axes[1])
    axes[1].text(0.0, 1.145, "(b)  Four ordinary tools lift every level",
                 transform=axes[1].transAxes, fontsize=10.5,
                 fontweight="bold", ha="left", va="bottom")
    axes[1].text(0.0, 1.105,
                 "L1–L3 signed chance-corrected · L4 uncorrected 0/0.5/1 rubric, "
                 "not comparable across the L4 boundary · 800-item subset",
                 transform=axes[1].transAxes, fontsize=7.6, color=STEEL,
                 ha="left", va="bottom", style="italic")
    fig.tight_layout(h_pad=1.9)
    out = pathlib.Path("output/figures/fig_agent_pipeline_results.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=170)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
