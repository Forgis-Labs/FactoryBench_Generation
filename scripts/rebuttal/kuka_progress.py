"""Print how many KUKA eval items are done per (model, level) cell.

Usage:  python scripts/rebuttal/kuka_progress.py
        python scripts/rebuttal/kuka_progress.py --root output/kuka_eval/replies
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


LEVEL_TOTAL = {"level1": 98, "level2": 346, "level3": 24, "level4": 119}

# display -> HF slug used in the reply directory name
MODEL_SLUGS = {
    "gpt-5.1-1":         "gpt-5_1-1",
    "claude-sonnet-4.6": "claude-sonnet-4_6",
    "mistral-large-3":   "mistral-large-3",
    "deepseek-v3.2":     "deepseek-v3_2",
    "qwen-3-235b":       "qwen-3-235b",
    "qwen-3-4b":         "qwen-3-4b",
}


def _count(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for name in os.listdir(d) if name.endswith("_answer.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("output/kuka_eval/replies"))
    args = ap.parse_args()

    per_level = sum(LEVEL_TOTAL.values())
    target = per_level * len(MODEL_SLUGS)

    header = f"{'model':<22}" + "".join(f"{f'{L}({n})':>12}" for L, n in LEVEL_TOTAL.items()) + f"   {'total':>12}"
    print(header)
    print("-" * len(header))

    grand_done = 0
    for display, slug in MODEL_SLUGS.items():
        cells = [_count(args.root / lvl / slug) for lvl in LEVEL_TOTAL]
        tot = sum(cells)
        grand_done += tot
        pct = 100 * tot / per_level
        row = f"{display:<22}" + "".join(f"{f'{c}/{LEVEL_TOTAL[lvl]}':>12}" for c, lvl in zip(cells, LEVEL_TOTAL)) + f"   {tot}/{per_level} ({pct:.0f}%)"
        print(row)

    print()
    print(f"overall: {grand_done}/{target} = {100 * grand_done / target:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
