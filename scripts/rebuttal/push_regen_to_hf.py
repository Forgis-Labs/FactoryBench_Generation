"""Push regenerated Q&A JSONLs to Hugging Face, one file at a time.

Properties (designed to be safe if interrupted mid-run):
  * Sequential uploads. Each HF upload is atomic on their side, so a
    completed push is durable even if the script is killed after.
  * ``.pushed`` marker file next to each source JSONL. Rerunning with
    ``--resume`` (default: on) skips files that have already been
    successfully pushed.
  * Per-file progress line: level+split, size in MB, wall-clock, and the
    running success/failure counters.

Usage:
    python scripts/rebuttal/push_regen_to_hf.py                    # push all 9
    python scripts/rebuttal/push_regen_to_hf.py --no-resume        # force repush everything
    python scripts/rebuttal/push_regen_to_hf.py --dry-run          # print plan, don't push
    python scripts/rebuttal/push_regen_to_hf.py --only 2 test      # push only level_2 test
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

REPO_ID = "FactoryBench/FactoryBench"
REPO_TYPE = "dataset"
LEVELS = [1, 2, 3]
SPLITS = ["train", "validation", "test"]


def _fmt_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, default=Path("output/regen_10hz"),
                    help="local directory with level_<L>_<split>.jsonl files")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore .pushed markers and push every file")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without actually uploading")
    ap.add_argument("--only", nargs=2, metavar=("LEVEL", "SPLIT"),
                    help='push only a single file, e.g. --only 2 test')
    ap.add_argument("--commit-message-prefix", type=str,
                    default="regen: fix impossible-at-10Hz windows",
                    help="commit message prefix appended with (L<L>/<split>)")
    args = ap.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    tok = os.environ["HF_WRITE_TOKEN"]

    from huggingface_hub import HfApi
    api = HfApi(token=tok)

    plan: list[tuple[int, str, Path]] = []
    if args.only:
        lvl = int(args.only[0]); split = args.only[1]
        plan.append((lvl, split, args.src_dir / f"level_{lvl}_{split}.jsonl"))
    else:
        for lvl in LEVELS:
            for split in SPLITS:
                plan.append((lvl, split, args.src_dir / f"level_{lvl}_{split}.jsonl"))

    # sanity-check every planned file exists locally
    missing = [p for _, _, p in plan if not p.exists()]
    if missing:
        print(f"missing local files, aborting:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 2

    print(f"plan: {len(plan)} file(s), source={args.src_dir}, resume={not args.no_resume}, dry_run={args.dry_run}")
    print()

    n_ok = 0; n_skipped = 0; n_failed = 0
    for i, (lvl, split, path) in enumerate(plan, 1):
        pushed_marker = path.with_suffix(path.suffix + ".pushed")
        remote = f"factorybench_qa/level_{lvl}/{split}.jsonl"
        size = path.stat().st_size
        prefix = f"[{i}/{len(plan)}] L{lvl}/{split}"
        if pushed_marker.exists() and not args.no_resume:
            n_skipped += 1
            print(f"{prefix}: SKIP (already pushed; delete {pushed_marker.name} to force)")
            continue
        if args.dry_run:
            print(f"{prefix}: WOULD PUSH {path} ({_fmt_mb(size)}) -> hf://{REPO_ID}/{remote}")
            continue
        t0 = time.time()
        try:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=remote,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=f"{args.commit_message_prefix} (L{lvl}/{split})",
            )
            pushed_marker.touch()
            dt = time.time() - t0
            n_ok += 1
            print(f"{prefix}: OK  {_fmt_mb(size)} in {dt:.1f}s "
                  f"({size / dt / (1024*1024):.1f} MB/s) -> hf://{remote}   "
                  f"[running: ok={n_ok} skipped={n_skipped} failed={n_failed}]",
                  flush=True)
        except Exception as e:
            dt = time.time() - t0
            n_failed += 1
            print(f"{prefix}: FAIL after {dt:.1f}s: {type(e).__name__}: {e}   "
                  f"[running: ok={n_ok} skipped={n_skipped} failed={n_failed}]",
                  flush=True)

    print()
    print(f"summary: ok={n_ok} skipped={n_skipped} failed={n_failed}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
