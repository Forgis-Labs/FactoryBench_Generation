"""Draw a stratified random subset of prompts from the full merged test tree.

Samples ``--per-level N`` items per level uniformly at random (seeded), copies
their prompt files into ``<out-dir>/level<N>/``. The agent runner then walks
that directory just like it would a full prompts tree — no code changes.

Sampling is stratified by level (not global-random) so we don't
undersample small levels (L3 = 321 items). Within a level the sample is
uniform random. The KUKA vs UR3 vs AURSAD vs voraus-AD mix inside each
level is preserved in proportion to the test split.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-root", type=Path, default=Path("output/test_eval/prompts"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--per-level", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4])
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total = 0
    for L in args.levels:
        src = args.prompts_root / f"level{L}"
        if not src.is_dir():
            print(f"L{L}: {src} not found; skipping", file=sys.stderr)
            continue
        prompts = sorted(src.glob("*.json"))
        k = min(args.per_level, len(prompts))
        picked = rng.sample(prompts, k)
        dst = args.out_dir / f"level{L}"
        dst.mkdir(parents=True, exist_ok=True)
        for p in picked:
            shutil.copy(p, dst / p.name)
        print(f"L{L}: picked {k}/{len(prompts)} → {dst}")
        total += k
    print(f"total: {total} prompts sampled with seed={args.seed} → {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
