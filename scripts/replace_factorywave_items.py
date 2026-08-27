"""Swap the factorywave-sourced released items for regenerated ones.

The generators changed substantially: option evidence is now checked against
the channels the context actually shows, comparison pairs are drawn by target
cell so no proposition is guessable from its prior, multi-select thresholds are
fitted per robot, exact-negation statements can no longer share an item, and
the context carries the channels the item's own fault is diagnosable from.
None of that reaches the release until the items are rebuilt.

Only items whose every source is factorywave (or simulations) are swapped.
aursad and vorausad items are left exactly as published, because regenerating
them needs those episodes normalized locally and that is a separate pass.

The swap is one for one. Each replaced item takes the position of the item it
replaces, in the same split file, and comes from the pool for the same (level,
template) pair. Split sizes and the per-level, per-template counts the paper
reports therefore do not move at all. The ids do change, and the ground truth
with them, so any score previously reported on a replaced item no longer
applies.

Supply is checked for every (level, template) before a single file is written.
A pool that runs dry mid-file would leave a split half swapped, which is worse
than not running.

Usage:
    python scripts/replace_factorywave_items.py --pools <dir> --workdir <dir>
    ... --push
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from huggingface_hub import HfApi, hf_hub_download

REPO = "FactoryBench/FactoryBench"
SPLITS = ("train", "validation", "test")
LEVELS = (1, 2, 3, 4)
# Sources whose episodes are available locally and can therefore be rebuilt.
REBUILDABLE = frozenset({"factorywave", "factorywave_kuka", "simulations"})


def item_sources(item: Dict[str, Any], node: Any = None, out: Optional[set] = None) -> set:
    """Every dataset named anywhere in the item's provenance.

    Recursive, because the shape is not uniform. Most templates put the source
    in a top-level ``dataset`` or ``dataset_a``, but the group templates record
    a list of per-episode dicts under ``episodes`` instead. Reading only the top
    level found nothing for those, and an item whose sources look empty is
    neither clearly rebuildable nor clearly not, so all 1,858 L2.9 items were
    being skipped when 811 of them are pure factorywave.
    """
    if out is None:
        out = set()
        node = item.get("provenance") or {}
    if isinstance(node, dict):
        for key, value in node.items():
            if (key == "dataset" or key.startswith("dataset_") or key == "cf_dataset") \
                    and isinstance(value, str) and value:
                out.add(value)
            else:
                item_sources(item, value, out)
    elif isinstance(node, list):
        for value in node:
            item_sources(item, value, out)
    return out


def is_rebuildable(item: Dict[str, Any]) -> bool:
    sources = item_sources(item)
    return bool(sources) and sources <= REBUILDABLE


_HEAD = re.compile(rb'"level"\s*:\s*(\d+).*?"template_id"\s*:\s*(\d+)', re.S)


def index_pools(pools_dir: Path) -> Dict[Tuple[int, int], List[Path]]:
    """(level, template_id) -> generated item paths, from every pool subdirectory.

    Paths, not parsed items. A full corpus regeneration produces tens of
    thousands of items each carrying its own time series, and holding them all
    in memory to swap a few of them in costs several GB for no reason. Each
    item is read once, at the moment it replaces something.

    Level and template sit near the front of every generated file, so the
    classification reads a fixed prefix rather than parsing the whole object.
    """
    pools: Dict[Tuple[int, int], List[Path]] = collections.defaultdict(list)
    for path in sorted(pools_dir.rglob("level*.json")):
        try:
            with path.open("rb") as fh:
                head = fh.read(512)
        except OSError:
            continue
        match = _HEAD.search(head)
        if match:
            pools[(int(match.group(1)), int(match.group(2)))].append(path)
    return pools


def read_item(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarise(answers: Iterable[Any]) -> str:
    """Answer-shape stats, enough to see a template is not degenerate."""
    answers = [a for a in answers if isinstance(a, str) and a]
    if not answers:
        return "no string answers"
    counter = collections.Counter(answers)
    top, n = counter.most_common(1)[0]
    out = f"{len(counter)} distinct, top {top!r} at {n / len(answers):.1%}"
    if all(len(a) == 4 and set(a) <= {"T", "F"} for a in answers):
        total = len(answers)
        majority = sum(
            max(sum(a[i] == "T" for a in answers), total - sum(a[i] == "T" for a in answers))
            for i in range(4)
        ) / (4 * total)
        out += f", majority-class {majority:.3f}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", type=Path, required=True,
                    help="Directory of generated items, searched recursively.")
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--token-env", default="HF_WRITE_TOKEN")
    ap.add_argument("--levels", type=int, nargs="+", default=list(LEVELS),
                    help="Only rebuild these levels (default: all). Levels left "
                         "out are not downloaded, spliced or staged at all.")
    args = ap.parse_args()
    levels = tuple(args.levels)

    token = os.getenv(args.token_env)
    if args.push and not token:
        raise SystemExit(f"{args.token_env} not set")
    api = HfApi(token=token) if token else None
    args.workdir.mkdir(parents=True, exist_ok=True)

    pools = index_pools(args.pools)
    print(f"pools indexed from {args.pools}:")
    for key in sorted(pools):
        print(f"  L{key[0]}.{key[1]:<3} {len(pools[key]):6d} items")
    print(f"  total {sum(len(v) for v in pools.values())}")

    # Demand first, across every split, so a short pool stops the run before
    # any file is written rather than halfway through one.
    downloads: Dict[Tuple[int, str], Path] = {}
    demand: collections.Counter = collections.Counter()
    for level in levels:
        for split in SPLITS:
            remote = f"factorybench_qa/level_{level}/{split}.jsonl"
            local = Path(hf_hub_download(REPO, remote, repo_type="dataset", token=token,
                                         local_dir=args.workdir / "download"))
            downloads[(level, split)] = local
            for line in local.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if is_rebuildable(item):
                    demand[(level, item.get("template_id"))] += 1

    print("\ndemand vs supply:")
    short = []
    for key in sorted(demand):
        have = len(pools.get(key, ()))
        flag = ""
        if have < demand[key]:
            short.append((key, demand[key], have))
            flag = "   SHORT"
        print(f"  L{key[0]}.{key[1]:<3} need {demand[key]:6d}  have {have:6d}{flag}")
    if short:
        print("\npools too small; generate more before swapping:")
        for key, need, have in short:
            print(f"  L{key[0]}.{key[1]} short by {need - have}")
        return 1

    cursors: Dict[Tuple[int, int], int] = collections.defaultdict(int)
    before: Dict[Tuple[int, int], List] = collections.defaultdict(list)
    after: Dict[Tuple[int, int], List] = collections.defaultdict(list)
    staged: List[Tuple[Path, str]] = []
    swapped = kept = 0
    print()
    for level in levels:
        for split in SPLITS:
            remote = f"factorybench_qa/level_{level}/{split}.jsonl"
            rows: List[Dict[str, Any]] = []
            n_swapped = 0
            for line in downloads[(level, split)].read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if is_rebuildable(item):
                    key = (level, item.get("template_id"))
                    before[key].append(item.get("answer"))
                    item = read_item(pools[key][cursors[key]])
                    cursors[key] += 1
                    after[key].append(item.get("answer"))
                    n_swapped += 1
                    swapped += 1
                else:
                    kept += 1
                rows.append(item)
            out = args.workdir / "swapped" / remote
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
            staged.append((out, remote))
            print(f"  L{level}/{split:11s} {len(rows):6d} rows, {n_swapped:6d} swapped")

    print(f"\nswapped {swapped}, kept {kept}, total {swapped + kept}")
    print("\nanswer shape, before -> after:")
    for key in sorted(before):
        print(f"  L{key[0]}.{key[1]:<3} before: {summarise(before[key])}")
        print(f"  {'':7s} after : {summarise(after[key])}")

    if not args.push:
        print("\ndry run, nothing pushed. Re-run with --push.")
        return 0
    for path, remote in staged:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=remote,
                        repo_id=REPO, repo_type="dataset",
                        commit_message="regenerate factorywave-sourced items")
        print(f"pushed {remote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
