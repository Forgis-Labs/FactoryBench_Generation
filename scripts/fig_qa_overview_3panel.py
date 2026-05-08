"""3-panel overview of the released FactoryBench Q&A dataset.

Loads the 12 split files (level_{1..4}/{train,validation,test}.jsonl) from
the public HuggingFace dataset and emits a single PDF with:

  (a) Dataset size per level (stacked train/validation/test).
  (b) Answer-format mix per level (test split, four canonical buckets).
  (c) Sub-series length distribution per level.

Output: docs/neurips_tex/figures/fig_qa_overview_3panel.pdf
"""
from __future__ import annotations

import json
import os
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv(r"C:\Users\ymerz\OneDrive\Documents\Work\Forgis\FactoryBench\.env",
            override=True)

REPO = pathlib.Path(__file__).resolve().parents[1]
HF_REPO = "FactoryBench/FactoryBench"
LEVELS = (1, 2, 3, 4)
SPLITS = ("train", "validation", "test")

OUT = REPO / "docs/neurips_tex/figures/fig_qa_overview_3panel.pdf"

TIGER    = "#ff5a00"
FLICKER  = "#DC4B07"
STEEL    = "#878f92"
GUNMETAL = "#122128"
LIGHT    = "#FFE2CC"

SPLIT_COLORS = {
    "train":      GUNMETAL,
    "validation": STEEL,
    "test":       TIGER,
}
LEVEL_COLORS = {
    1: "#3b82f6",
    2: TIGER,
    3: "#10b981",
    4: FLICKER,
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":         12,
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "legend.fontsize":   11,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "axes.linewidth":    0.9,
        "axes.edgecolor":    GUNMETAL,
        "axes.labelcolor":   GUNMETAL,
        "xtick.color":       GUNMETAL,
        "ytick.color":       GUNMETAL,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def _classify_answer_style(rec: dict) -> str:
    """Map a Q&A record to one of: numerical, numerical_tensor, mc_single_K,
    mc_multi_4, free_form."""
    options = rec.get("options") or {}
    answer = rec.get("answer")
    if isinstance(answer, list):
        return "numerical_tensor"
    if not options:
        if isinstance(answer, (int, float)):
            return "numerical"
        return "free_form"
    n_opts = len(options)
    # Multi-select answers tend to be a string of T/F flags or a list of keys.
    if isinstance(answer, str) and re.fullmatch(r"[TF]+", answer):
        return f"mc_multi_{len(answer)}"
    if isinstance(answer, list):
        return f"mc_multi_{n_opts}"
    return f"mc_single_{n_opts}"


def load_all() -> pd.DataFrame:
    token = os.getenv("HF_TOKEN")
    rows = []
    for level in LEVELS:
        for split in SPLITS:
            path = f"factorybench_qa/level_{level}/{split}.jsonl"
            local = hf_hub_download(
                repo_id=HF_REPO, filename=path, repo_type="dataset",
                token=token, force_download=False,
            )
            with open(local, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    ts = rec.get("context", {}).get("time_series", [])
                    rows.append({
                        "level":        level,
                        "split":        split,
                        "id":           rec.get("id"),
                        "answer_style": _classify_answer_style(rec),
                        "ts_rows":      len(ts) if isinstance(ts, list) else 0,
                    })
    return pd.DataFrame(rows)


def main():
    set_style()
    df = load_all()
    print(f"loaded {len(df):,} Q&A items")

    fig, axes = plt.subplots(3, 1, figsize=(10, 12.5))
    plt.subplots_adjust(hspace=0.45)

    # (a) split sizes per level
    ax = axes[0]
    pivot = (
        df.pivot_table(index="level", columns="split", values="id", aggfunc="count")
        .fillna(0)
        .astype(int)
    )
    pivot = pivot[[s for s in SPLITS if s in pivot.columns]]
    bottom = np.zeros(len(pivot))
    for split in pivot.columns:
        vals = pivot[split].values
        ax.bar(pivot.index, vals, bottom=bottom, color=SPLIT_COLORS[split],
               label=split, edgecolor="white", linewidth=0.6)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(pivot.index[i], bottom[i] + v / 2, f"{int(v):,}",
                        ha="center", va="center", color="white",
                        fontsize=11, fontweight="bold")
        bottom += vals
    ax.set_xticks(list(pivot.index))
    ax.set_xticklabels([f"L{lvl}" for lvl in pivot.index])
    ax.set_ylabel("Samples")
    ax.legend(title="Split", loc="upper right", frameon=False)
    ax.set_title(f"(a) Dataset size per level  ({pivot.values.sum():,} samples total)")

    # (b) answer-format mix per level (test split, four canonical buckets)
    STYLE_BUCKET = {
        "numerical":        "Numerical / Tensor",
        "numerical_tensor": "Numerical / Tensor",
        "mc_single_3":      "MC single-select",
        "mc_single_4":      "MC single-select",
        "mc_multi_4":       "MC multi-select",
        "free_form":        "Free-form",
    }
    BUCKET_ORDER = [
        "Numerical / Tensor",
        "MC single-select",
        "MC multi-select",
        "Free-form",
    ]
    BUCKET_COLOR = {
        "Numerical / Tensor": TIGER,
        "MC single-select":   "#3b82f6",
        "MC multi-select":    FLICKER,
        "Free-form":          GUNMETAL,
    }

    ax = axes[1]
    test_only = df[df.split == "test"].copy()
    test_only["bucket"] = test_only["answer_style"].map(STYLE_BUCKET).fillna("Other")
    afmt = (
        test_only.pivot_table(index="bucket", columns="level",
                              values="id", aggfunc="count")
        .fillna(0).astype(int)
    )
    afmt = afmt.reindex(index=[b for b in BUCKET_ORDER if b in afmt.index])
    proportions = afmt.div(afmt.sum(axis=0), axis=1).fillna(0)
    bottom = np.zeros(proportions.shape[1])
    for bucket in proportions.index:
        vals = proportions.loc[bucket].values
        color = BUCKET_COLOR.get(bucket, STEEL)
        ax.bar([f"L{lvl}" for lvl in proportions.columns], vals, bottom=bottom,
               color=color, label=bucket, edgecolor="white", linewidth=0.5)
        for j, v in enumerate(vals):
            if v >= 0.06:
                ax.text(j, bottom[j] + v / 2, f"{int(round(v*100))}%",
                        ha="center", va="center", color="white",
                        fontsize=11, fontweight="bold")
        bottom += vals
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(p*100)}%" for p in np.linspace(0, 1, 6)])
    ax.legend(title="Answer format", bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=False)
    ax.set_title("(b) Answer-format mix (test split)")

    # (c) sub-series length distribution
    ax = axes[2]
    for lvl in LEVELS:
        sub = df[df.level == lvl]["ts_rows"]
        sub = sub[sub > 0]
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=30, alpha=0.55, color=LEVEL_COLORS[lvl],
                label=f"L{lvl}  median={int(sub.median())}",
                edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Time-series rows per sample")
    ax.set_ylabel("Samples")
    ax.legend(frameon=False)
    ax.set_title("(c) Sub-series length distribution")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
