"""Rescore ALL model replies by re-parsing each reply's `answer` text against
the (possibly updated) question `acceptance_bounds`, then applying the signed
chance correction. Unlike `rescore_signed.py`, this ignores the reply's cached
`score` field — needed after the template-7 margin patch.

Only re-scores the numerical branch (which is where the template-7 bug was);
other formats fall back to the cached score so free-form judgements stay intact.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Local imports — assume PYTHONPATH=. from repo root.
from src.evaluation.chance_correct import chance_correct
from src.evaluation.run_foundry_eval import score_prediction


def _load_question(questions_root: Path, level: str, custom_id: str) -> Optional[Dict[str, Any]]:
    """Match reply custom_id to a question file across the naming variants seen."""
    stem_candidates: List[str] = []
    stripped = custom_id
    parts = custom_id.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        stripped = "_".join(parts[:-1])
    stem_candidates.extend([stripped, custom_id])
    # also try level-prefixed and un-prefixed
    for c in list(stem_candidates):
        if c.startswith(f"{level}_"):
            stem_candidates.append(c[len(level) + 1:])
        else:
            stem_candidates.append(f"{level}_{c}")
    for stem in stem_candidates:
        p = questions_root / level / f"{stem}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies-root", type=Path, required=True)
    ap.add_argument("--questions-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--levels", nargs="*", default=None,
                    help="restrict to these level dirs (e.g. level1)")
    args = ap.parse_args()

    buckets: Dict[tuple, Dict[str, List[float]]] = defaultdict(
        lambda: {"raw": [], "signed": [], "clipped": []}
    )
    n_files = n_rescored_num = n_used_cached = n_no_q = 0
    for level_dir in sorted(args.replies_root.iterdir()):
        if not level_dir.is_dir(): continue
        level = level_dir.name
        if args.levels and level not in args.levels: continue
        for model_dir in sorted(level_dir.iterdir()):
            if not model_dir.is_dir(): continue
            model = model_dir.name
            for rp in model_dir.glob("*_answer.json"):
                n_files += 1
                try:
                    rep = json.loads(rp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                cid = rep.get("custom_id") or rp.stem[:-len("_answer")]
                q = _load_question(args.questions_root, level, cid)
                if q is None:
                    n_no_q += 1
                    continue
                fmt = rep.get("answer_format") or q.get("answer_format")
                if not fmt: continue
                if fmt == "free_form":
                    continue
                if fmt == "numerical":
                    # re-score against fresh bounds
                    try:
                        raw_score, _ = score_prediction(
                            answer_format=fmt,
                            prediction=rep.get("answer") or "",
                            ground_truth=q.get("answer"),
                            acceptance_bounds=q.get("acceptance_bounds"),
                            question_text=rep.get("prompt") or "",
                            judge_model="",
                        )
                    except Exception:
                        raw_score = None
                    if raw_score is None:
                        raw_score = rep.get("score")
                        if raw_score is None: continue
                    else:
                        n_rescored_num += 1
                else:
                    raw_score = rep.get("score")
                    if raw_score is None: continue
                    n_used_cached += 1
                signed = chance_correct(raw_score, fmt, q, clip=False)
                clipped = chance_correct(raw_score, fmt, q, clip=True)
                key = (level, model)
                buckets[key]["raw"].append(float(raw_score))
                if signed is not None:  buckets[key]["signed"].append(signed)
                if clipped is not None: buckets[key]["clipped"].append(clipped)

    rows = []
    for (level, model), b in sorted(buckets.items()):
        row = {
            "level": level, "model": model, "n": len(b["raw"]),
            "raw_mean":    statistics.fmean(b["raw"])    if b["raw"] else None,
            "signed_mean": statistics.fmean(b["signed"]) if b["signed"] else None,
            "clipped_mean":statistics.fmean(b["clipped"])if b["clipped"] else None,
        }
        rows.append(row)
    args.out.write_text(json.dumps({
        "rows": rows,
        "n_files": n_files,
        "n_rescored_numerical": n_rescored_num,
        "n_used_cached_score": n_used_cached,
        "n_no_question": n_no_q,
    }, indent=2), encoding="utf-8")

    # print
    print(f"scanned {n_files} replies; rescored {n_rescored_num} numerical items; "
          f"cached-score {n_used_cached}; missing questions {n_no_q}")
    print(f"{'level':<8s} {'model':<25s} {'n':>5s}  {'raw%':>7s} {'signed%':>7s} {'clip%':>7s}")
    for r in rows:
        print(f"{r['level']:<8s} {r['model']:<25s} {r['n']:>5d}  "
              f"{r['raw_mean']*100 if r['raw_mean'] is not None else 0:7.2f} "
              f"{r['signed_mean']*100 if r['signed_mean'] is not None else 0:7.2f} "
              f"{r['clipped_mean']*100 if r['clipped_mean'] is not None else 0:7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
