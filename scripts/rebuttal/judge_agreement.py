#!/usr/bin/env python
"""Inter-judge reliability analysis for the FactoryBench Level-4 LLM-as-judge panel.

Every L4 answer JSON stores the three judges' verdicts under `llm_judge_votes`
(gpt-5.1, claude-sonnet-4.6, deepseek-v3.2), each in {0, 0.5, 1}. We pool the
judged answers across the evaluee panel (the judges' agreement is a property of
the judges, not of the model being scored) and compute:

  * per-judge score distribution + mean (leniency/strictness bias),
  * pairwise exact agreement and quadratic-weighted Cohen's kappa,
  * pairwise 3x3 confusion matrices,
  * 3-way Fleiss' kappa and Krippendorff's alpha (interval),
  * consensus breakdown (unanimous / 2-1 majority / three-way split),
  * the same, split by question type (troubleshooting vs optimization).

Outputs results.json used by the figure script.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

# evaluee panel whose L4 answers we pool (paper panel; exclude finetuned/empty dirs)
PANEL = ["claude-sonnet-4_6", "deepseek-v3_2", "gpt-5_1-1",
         "mistral-large-3", "qwen-3-235b", "qwen-3-4b"]
JUDGES = ["gpt-5.1-1", "claude-sonnet-4.6", "deepseek-v3.2"]
JUDGE_SHORT = {"gpt-5.1-1": "GPT-5.1", "claude-sonnet-4.6": "Claude 4.6",
               "deepseek-v3.2": "DeepSeek"}
LABELS = [0.0, 0.5, 1.0]


def load_votes(repo: Path, level_dir: str):
    """Return list of rows: {judge: score} + question_type, for items with >=1 vote."""
    root = repo / "output" / "test_eval" / "replies" / level_dir
    rows = []
    for model in PANEL:
        mdir = root / model
        if not mdir.is_dir():
            continue
        for f in mdir.glob("*_answer.json"):
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            votes = d.get("llm_judge_votes") or {}
            if not votes:
                continue
            row = {"model": model, "qtype": d.get("question_type", "")}
            for j in JUDGES:
                v = votes.get(j) or {}
                s = v.get("score")
                row[j] = float(s) if s is not None else None
            rows.append(row)
    return rows


def cohen_quadratic(a, b):
    from sklearn.metrics import cohen_kappa_score
    # map {0,0.5,1} -> integer categories {0,1,2}; equal spacing so quadratic
    # weights are unchanged. sklearn's kappa rejects continuous labels.
    idx = {v: i for i, v in enumerate(LABELS)}
    ai, bi = [idx[x] for x in a], [idx[y] for y in b]
    return float(cohen_kappa_score(ai, bi, weights="quadratic", labels=[0, 1, 2]))


def fleiss_kappa(rating_rows):
    """rating_rows: list of 3-tuples of category scores. Returns Fleiss' kappa."""
    n_items = len(rating_rows)
    n_raters = 3
    idx = {v: i for i, v in enumerate(LABELS)}
    N = np.zeros((n_items, len(LABELS)))
    for r, row in enumerate(rating_rows):
        for s in row:
            N[r, idx[s]] += 1
    p_j = N.sum(0) / (n_items * n_raters)          # category marginals
    P_i = (np.square(N).sum(1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e = np.square(p_j).sum()
    return float((P_bar - P_e) / (1 - P_e)) if (1 - P_e) > 0 else float("nan")


def krippendorff_alpha_interval(rating_rows):
    """Interval-metric Krippendorff's alpha for the 3-rater complete design.

    Uses the identity sum_{i!=j}(v_i - v_j)^2 = 2*(m*sum(v^2) - (sum v)^2) so both
    observed (Do) and expected (De) disagreement are O(m), not O(m^2).
    """
    vals = np.array(rating_rows, dtype=float)          # (n_items, 3)
    n = vals.shape[1]
    # observed: per-item pairwise squared diffs, summed, normalised
    rowsum = vals.sum(1)
    rowsq = (vals ** 2).sum(1)
    Do = (2.0 * (n * rowsq - rowsum ** 2)).sum() / (len(vals) * n * (n - 1))
    # expected: over all values pooled
    allv = vals.ravel()
    m = allv.size
    s1, s2 = allv.sum(), (allv ** 2).sum()
    De = 2.0 * (m * s2 - s1 ** 2) / (m * (m - 1))
    return float(1 - Do / De) if De > 0 else float("nan")


def analyse(rows, tag):
    # complete cases: all three judges returned a score
    complete = [r for r in rows if all(r[j] is not None for j in JUDGES)]
    res = {"tag": tag, "n_total": len(rows), "n_complete": len(complete)}

    # per-judge distribution + mean (on complete cases for comparability)
    dist = {}
    for j in JUDGES:
        c = Counter(r[j] for r in complete)
        tot = sum(c.values())
        dist[JUDGE_SHORT[j]] = {
            "frac": {str(l): c.get(l, 0) / tot for l in LABELS},
            "mean": float(np.mean([r[j] for r in complete])),
        }
    res["per_judge"] = dist

    # pairwise agreement, weighted kappa, confusion matrices
    pair = {}
    for ja, jb in itertools.combinations(JUDGES, 2):
        a = [r[ja] for r in complete]
        b = [r[jb] for r in complete]
        exact = float(np.mean([x == y for x, y in zip(a, b)]))
        kappa = cohen_quadratic(a, b)
        cm = np.zeros((3, 3), dtype=int)
        idx = {v: i for i, v in enumerate(LABELS)}
        for x, y in zip(a, b):
            cm[idx[x], idx[y]] += 1
        pair[f"{JUDGE_SHORT[ja]} vs {JUDGE_SHORT[jb]}"] = {
            "exact_agreement": exact, "weighted_kappa": kappa,
            "confusion": cm.tolist(),
            "a": JUDGE_SHORT[ja], "b": JUDGE_SHORT[jb],
        }
    res["pairwise"] = pair

    # 3-way reliability
    triples = [tuple(r[j] for j in JUDGES) for r in complete]
    res["fleiss_kappa"] = fleiss_kappa(triples)
    res["krippendorff_alpha"] = krippendorff_alpha_interval(triples)

    # consensus breakdown
    cons = Counter()
    for t in triples:
        c = Counter(t)
        top = max(c.values())
        cons["unanimous" if top == 3 else "majority" if top == 2 else "split"] += 1
    tot = sum(cons.values())
    res["consensus"] = {k: cons.get(k, 0) / tot for k in ("unanimous", "majority", "split")}
    res["consensus_counts"] = dict(cons)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("output/judge_agreement"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_votes(args.repo, "level4")
    print(f"loaded {len(rows)} judged L4 answers across {len(PANEL)} evaluees")
    out = {"overall": analyse(rows, "all")}
    for qt in ("troubleshooting", "optimization"):
        sub = [r for r in rows if r["qtype"] == qt]
        if sub:
            out[qt] = analyse(sub, qt)

    json.dump(out, open(args.out / "results.json", "w"), indent=2)
    o = out["overall"]
    print(f"\ncomplete cases (3 valid votes): {o['n_complete']}")
    print("per-judge mean:", {k: round(v["mean"], 3) for k, v in o["per_judge"].items()})
    print("Fleiss kappa:", round(o["fleiss_kappa"], 3),
          "| Krippendorff alpha:", round(o["krippendorff_alpha"], 3))
    for name, p in o["pairwise"].items():
        print(f"  {name}: exact={p['exact_agreement']:.3f} wkappa={p['weighted_kappa']:.3f}")
    print("consensus:", {k: round(v, 3) for k, v in o["consensus"].items()})
    print(f"\nwrote {args.out/'results.json'}")


if __name__ == "__main__":
    main()
