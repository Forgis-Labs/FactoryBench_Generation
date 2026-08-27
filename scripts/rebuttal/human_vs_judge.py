#!/usr/bin/env python
"""Compare an independent grader's scores to the LLM-judge panel on the 100-item
Level-4 sample. Reads output/human_baseline/{sample.json, independent_scores.json}
where independent_scores.json is {review_id(str): score in {0,0.5,1}}."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

D = Path("output/human_baseline")
sample = {r["review_id"]: r for r in json.load(open(D / "sample.json", encoding="utf-8"))}
indep = json.load(open(D / "independent_scores.json", encoding="utf-8"))

rows = []
for rid, r in sample.items():
    if str(rid) not in indep:
        continue
    j = r["judges"]
    rows.append({"rid": rid, "qtype": r["question_type"], "indep": float(indep[str(rid)]),
                 "median": r["median"], "GPT-5.1": j["GPT-5.1"]["score"],
                 "Claude 4.6": j["Claude 4.6"]["score"], "DeepSeek": j["DeepSeek"]["score"]})

IDX = {0.0: 0, 0.5: 1, 1.0: 2}


def agree(a, b):
    a, b = list(a), list(b)
    exact = float(np.mean([x == y for x, y in zip(a, b)]))
    wk = float(cohen_kappa_score([IDX[x] for x in a], [IDX[x] for x in b],
                                 weights="quadratic", labels=[0, 1, 2]))
    return exact, wk


ind = [r["indep"] for r in rows]
med = [r["median"] for r in rows]
e, k = agree(ind, med)
print(f"n={len(rows)} scored")
print(f"INDEPENDENT vs judge-median:  exact={e:.3f}  quad-weighted kappa={k:.3f}")
for jn in ("GPT-5.1", "Claude 4.6", "DeepSeek"):
    e2, k2 = agree(ind, [r[jn] for r in rows])
    print(f"  vs {jn:11s}: exact={e2:.3f}  wk={k2:.3f}")
lower = sum(r["indep"] < r["median"] for r in rows)
higher = sum(r["indep"] > r["median"] for r in rows)
print(f"\nindependent STRICTER than panel on {lower}, MORE LENIENT on {higher}, "
      f"agree on {len(rows)-lower-higher}")
print(f"mean score: independent={np.mean(ind):.3f}  judge-median={np.mean(med):.3f}")
for qt in ("troubleshooting", "optimization"):
    sub = [r for r in rows if r["qtype"] == qt]
    if sub:
        e3, k3 = agree([r["indep"] for r in sub], [r["median"] for r in sub])
        print(f"  {qt:15s} (n={len(sub)}): exact={e3:.3f} wk={k3:.3f}")
# confusion independent(rows) x median(cols)
print("\nconfusion  indep\\median   0     0.5    1")
cm = Counter((r["indep"], r["median"]) for r in rows)
for a in (0.0, 0.5, 1.0):
    print(f"   {a:>3}            " + "  ".join(f"{cm.get((a,b),0):4d}" for b in (0.0, 0.5, 1.0)))
json.dump({"n": len(rows), "exact_vs_median": e, "wkappa_vs_median": k,
           "mean_indep": float(np.mean(ind)), "mean_median": float(np.mean(med)),
           "stricter": lower, "lenient": higher},
          open(D / "human_vs_judge_summary.json", "w"), indent=2)
