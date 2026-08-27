"""Upload already-staged split files to the Hub.

`replace_factorywave_items.py` writes the finished splits under
`<workdir>/swapped/` during its dry run. Re-running it with `--push` would
redo the whole job, re-indexing tens of thousands of pool files and re-splicing
every split, only to upload the same bytes it already produced. This uploads
what is on disk.

It re-derives the counts from the staged files themselves rather than trusting
the dry run's report, and refuses to upload a split whose row count does not
match what is currently published. A split that gained or lost rows means the
splice went wrong, and that is exactly the case where uploading is worst.

Usage:
    python scripts/upload_staged_qa.py --staged <workdir>/swapped
    python scripts/upload_staged_qa.py --staged <workdir>/swapped --push
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from huggingface_hub import HfApi, hf_hub_download

REPO = "FactoryBench/FactoryBench"
SPLITS = ("train", "validation", "test")
LEVELS = (1, 2, 3, 4)


def load_env(path: Path) -> None:
    """Read KEY=VALUE lines into the environment without overwriting."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staged", type=Path, required=True,
                    help="Directory holding factorybench_qa/level_*/*.jsonl")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--token-env", default="HF_WRITE_TOKEN")
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    ap.add_argument("--message", default="regenerate factorywave-sourced Q&A items")
    ap.add_argument("--levels", type=int, nargs="+", default=list(LEVELS),
                    help="Only upload these levels (default: all). Use it to hold "
                         "back a level whose regeneration is not trusted yet.")
    args = ap.parse_args()

    load_env(args.env_file)
    token = os.getenv(args.token_env)
    if args.push and not token:
        raise SystemExit(f"{args.token_env} not found in the environment or {args.env_file}")

    plan: List[Tuple[Path, str, int, int]] = []
    mismatched = []
    for level in args.levels:
        for split in SPLITS:
            remote = f"factorybench_qa/level_{level}/{split}.jsonl"
            local = args.staged / remote
            if not local.is_file():
                raise SystemExit(f"missing staged file: {local}")
            published = Path(hf_hub_download(REPO, remote, repo_type="dataset", token=token))
            new_rows, old_rows = count_rows(local), count_rows(published)
            plan.append((local, remote, new_rows, old_rows))
            if new_rows != old_rows:
                mismatched.append((remote, old_rows, new_rows))

    print(f"{'file':44s} {'published':>10} {'staged':>8}  {'size':>9}")
    total_new = total_old = 0
    for local, remote, new_rows, old_rows in plan:
        total_new += new_rows
        total_old += old_rows
        flag = "" if new_rows == old_rows else "   ROW COUNT CHANGED"
        print(f"  {remote:42s} {old_rows:10d} {new_rows:8d}  "
              f"{local.stat().st_size / 1e6:8.1f}M{flag}")
    print(f"  {'TOTAL':42s} {total_old:10d} {total_new:8d}")

    if mismatched:
        print("\nrefusing to upload, these splits changed size:")
        for remote, old_rows, new_rows in mismatched:
            print(f"  {remote}: {old_rows} -> {new_rows}")
        return 1

    if not args.push:
        print("\ndry run, nothing uploaded. Re-run with --push.")
        return 0

    api = HfApi(token=token)
    for local, remote, _new, _old in plan:
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=REPO, repo_type="dataset", commit_message=args.message)
        print(f"pushed {remote}")
    print(f"\nuploaded {len(plan)} files, {total_new} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
