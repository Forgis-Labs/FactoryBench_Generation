#!/usr/bin/env python
"""Independent grader (held-out 4th rater) vs. the 3-judge panel on the 100-item
L4 sample. Emits fig_human_vs_judge.pdf and prints the stats."""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.metrics import cohen_kappa_score

ORANGE = "#FF5A00"; INK = "#122128"; STEEL = "#878F92"; BLUE = "#3A7BD5"; GREEN = "#2ECC71"
D = Path("output/human_baseline")
sample = {r["review_id"]: r for r in json.load(open(D / "sample.json", encoding="utf-8"))}
indep = json.load(open(D / "independent_scores.json"))
IDX = {0.0: 0, 0.5: 1, 1.0: 2}


def agree(a, b):
    exact = float(np.mean([x == y for x, y in zip(a, b)]))
    wk = float(cohen_kappa_score([IDX[x] for x in a], [IDX[x] for x in b],
                                 weights="quadratic", labels=[0, 1, 2]))
    return exact, wk


rows = [{"ind": float(indep[str(i)]), "median": r["median"], "qtype": r["question_type"],
         "GPT-5.1": r["judges"]["GPT-5.1"]["score"],
         "Claude 4.6": r["judges"]["Claude 4.6"]["score"],
         "DeepSeek": r["judges"]["DeepSeek"]["score"]}
        for i, r in sample.items() if str(i) in indep]
ind = [r["ind"] for r in rows]
raters = ["GPT-5.1", "Claude 4.6", "DeepSeek", "median"]
ex, wk, mean = {}, {}, {}
for name in raters:
    col = [r["median" if name == "median" else name] for r in rows]
    ex[name], wk[name] = agree(ind, col)
    mean[name] = float(np.mean(col))
mean["independent"] = float(np.mean(ind))

print(f"n={len(rows)}  independent-grader mean={mean['independent']:.3f}")
for name in raters:
    print(f"  vs {name:11s}: exact={ex[name]:.3f} wkappa={wk[name]:.3f} (mean {mean[name]:.3f})")

# ---- figure ----
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "cm",
                     "axes.edgecolor": INK, "axes.linewidth": 0.7, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7),
                       gridspec_kw={"width_ratios": [1.25, 0.95, 1.1]})
fig.subplots_adjust(left=0.065, right=0.99, top=0.85, bottom=0.24, wspace=0.42)

# (a) agreement of independent grader with each judge and the median
a = ax[0]; x = np.arange(len(raters)); w = 0.38
a.bar(x - w/2, [ex[n] for n in raters], w, color=ORANGE, edgecolor="white", lw=0.6, label="exact agreement")
a.bar(x + w/2, [wk[n] for n in raters], w, color=INK, edgecolor="white", lw=0.6, label="quad-weighted $\\kappa$")
for xi, n in enumerate(raters):
    a.text(xi - w/2, ex[n] + 0.015, f"{ex[n]:.2f}", ha="center", va="bottom", fontsize=6.0, color=ORANGE)
    a.text(xi + w/2, wk[n] + 0.015, f"{wk[n]:.2f}", ha="center", va="bottom", fontsize=6.0, color=INK)
a.set_xticks(x); a.set_xticklabels(["GPT-5.1", "Claude", "DeepSeek", "panel\nmedian"], fontsize=6.6)
a.set_ylim(0, 1.0); a.set_ylabel("agreement with\nindependent grader")
a.set_title("(a) Independent grader vs. each rater", fontsize=8.0)
a.legend(fontsize=6.0, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False, handlelength=1.0)

# (b) confusion independent x median
b = ax[1]
cm = np.zeros((3, 3))
for r in rows:
    cm[IDX[r["ind"]], IDX[r["median"]]] += 1
frac = cm / cm.sum()
cmap = LinearSegmentedColormap.from_list("f", ["#FFFFFF", "#FFD9C2", ORANGE])
b.imshow(frac, cmap=cmap, vmin=0, vmax=frac.max())
for i in range(3):
    for j in range(3):
        b.text(j, i, f"{int(cm[i,j])}", ha="center", va="center", fontsize=7.5,
               color="white" if frac[i, j] > frac.max()*0.6 else INK)
b.set_xticks(range(3)); b.set_xticklabels(["0", "0.5", "1"]); b.set_yticks(range(3)); b.set_yticklabels(["0", "0.5", "1"])
b.set_xlabel("panel median"); b.set_ylabel("independent grader")
b.set_title(f"(b) Confusion (exact {ex['median']*100:.0f}%)", fontsize=8.0)
for s in b.spines.values():
    s.set_visible(False)
b.tick_params(length=0)

# (c) mean score placement (leniency) on the 100-item sample
c = ax[2]
order = sorted(["DeepSeek", "GPT-5.1", "Claude 4.6", "median", "independent"],
               key=lambda n: mean[n])
short = {"DeepSeek": "DeepSeek", "GPT-5.1": "GPT-5.1", "Claude 4.6": "Claude",
         "median": "median", "independent": "indep."}
cols = [GREEN if n == "independent" else INK if n == "median" else STEEL for n in order]
vals = [mean[n] for n in order]
c.bar(range(len(order)), vals, 0.6, color=cols, edgecolor="white", lw=0.6)
for i, (n, v) in enumerate(zip(order, vals)):
    c.text(i, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=6.2, color=INK)
c.set_xticks(range(len(order)))
c.set_xticklabels([short[n] for n in order], fontsize=6.4, rotation=12)
c.set_ylim(0, max(vals)*1.25); c.set_ylabel("mean score (n=100)")
c.set_title("(c) Leniency placement", fontsize=8.0)

out = Path("output/figures/fig_human_vs_judge.pdf")
fig.savefig(out); plt.close(fig)
print("wrote", out)
json.dump({"n": len(rows), "exact": ex, "wkappa": wk, "mean": mean},
          open(D / "human_vs_judge_full.json", "w"), indent=2)
