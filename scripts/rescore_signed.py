"""Rescore already-saved model replies under the signed chance-correction rule.

The reviewer for W.1 correctly pointed out that the historical FactoryBench metric,
    s_tilde = max(0, (s - E)/(1 - E)),
does not average to 0 under uniform random guessing, because the max clip
zeroes out below-chance mass before averaging. The signed variant,
    s_tilde = (s - E)/(1 - E),
does average to exactly 0 under uniform random guessing by linearity of expectation.

Every reply JSON already stores the raw `score`, the `answer_format`, and a
link to its source question, so we can produce the corrected numbers without
re-running any LLM. This script walks `output/replies/level*/model*/` , applies
both the clipped and the signed correction, and prints per-level, per-model
means. Free-form items are excluded from the correction (E=0 by construction).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from src.evaluation.chance_correct import chance_correct


def _load_question(question_root: Path, custom_id: str) -> dict | None:
    # custom_id looks like "level1_1234_0" -> file "level1_1234.json"
    parts = custom_id.split("_")
    if len(parts) < 2:
        return None
    level = parts[0]
    stem = "_".join(parts[:-1])
    p = question_root / level / f"{stem}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies-root", default="output/replies")
    ap.add_argument("--questions-root", default="output/questions")
    ap.add_argument("--out", default="output/rescored_signed.json")
    args = ap.parse_args()

    replies_root = Path(args.replies_root)
    questions_root = Path(args.questions_root)

    # {(level, model): {"clipped": [...], "signed": [...], "raw": [...]}}
    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"clipped": [], "signed": [], "raw": []}
    )
    format_counts: dict[str, int] = defaultdict(int)

    n_files = 0
    n_no_question = 0
    for level_dir in sorted(replies_root.iterdir()):
        if not level_dir.is_dir() or not level_dir.name.startswith("level"):
            continue
        level = level_dir.name
        for model_dir in sorted(level_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for reply_path in model_dir.glob("*_answer.json"):
                try:
                    with open(reply_path, encoding="utf-8") as fh:
                        rep = json.load(fh)
                except Exception:
                    continue
                n_files += 1
                score = rep.get("score")
                if score is None:
                    continue
                fmt = rep.get("answer_format") or "free_form"
                format_counts[fmt] += 1
                if fmt == "free_form":
                    continue
                question = _load_question(questions_root, rep.get("custom_id", ""))
                if question is None:
                    n_no_question += 1
                clipped = chance_correct(score, fmt, question, clip=True)
                signed = chance_correct(score, fmt, question, clip=False)
                buckets[(level, model)]["raw"].append(float(score))
                if clipped is not None:
                    buckets[(level, model)]["clipped"].append(clipped)
                if signed is not None:
                    buckets[(level, model)]["signed"].append(signed)

    # Summarise
    rows = []
    for (level, model), b in sorted(buckets.items()):
        n = len(b["clipped"])
        row = {
            "level": level,
            "model": model,
            "n": n,
            "raw_mean": statistics.fmean(b["raw"]) if b["raw"] else None,
            "clipped_mean": statistics.fmean(b["clipped"]) if b["clipped"] else None,
            "signed_mean": statistics.fmean(b["signed"]) if b["signed"] else None,
        }
        rows.append(row)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "rows": rows,
                "format_counts": format_counts,
                "n_reply_files_scanned": n_files,
                "n_reply_missing_question": n_no_question,
            },
            fh,
            indent=2,
        )

    # Print a compact per-level table
    print(f"scanned {n_files} reply files")
    print(f"format counts: {dict(format_counts)}")
    print()
    print(f"{'level':<8}{'model':<25}{'n':>7}{'raw':>10}{'clipped':>10}{'signed':>10}")
    for r in rows:
        raw = f"{r['raw_mean']*100:>9.2f}" if r["raw_mean"] is not None else "     nan"
        clip = f"{r['clipped_mean']*100:>9.2f}" if r["clipped_mean"] is not None else "     nan"
        sig = f"{r['signed_mean']*100:>9.2f}" if r["signed_mean"] is not None else "     nan"
        print(f"{r['level']:<8}{r['model']:<25}{r['n']:>7d}{raw}{clip}{sig}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
