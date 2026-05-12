"""Partition the L1-L4 Q&A on HF into train/val/test by episode.

Splits are **episode-level and global**: a deterministic hash of the episode
stem decides the split, and every level uses the same mapping. So an episode
appearing in L1 and L2 lands in the same split for both — no cross-level
leakage. Default ratio: 80/10/10 with ``seed=42``.

Multi-episode questions (paired comparative L1 t3 / L2 t8; severity-ranking
L2 t9; L4 t3/t4 if enabled) only stay if **all** their episodes hash to the
same split. Otherwise they're dropped — keeps the splits hermetic.

Existing shards (``level_<N>/level<N>_shard_NNNN.jsonl``) are replaced by
``level_<N>/train.jsonl``, ``level_<N>/val.jsonl``, ``level_<N>/test.jsonl``.

Usage::

    python -m scripts.split_qa_train_val_test \\
        --dataset-folder factorynet_qa_150k \\
        --levels 1 2 3 4 \\
        --train-frac 0.8 --val-frac 0.1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from src.pipeline.upload_qa_pairs import DEFAULT_REPO_ID, resolve_hf_token

logger = logging.getLogger(__name__)


SHARD_RE = re.compile(r"level\d+_shard_\d+\.jsonl$")


def _split_for_episode(stem: str, seed: int, train_frac: float, val_frac: float) -> str:
    """Deterministic hash episode_stem -> 'train' | 'val' | 'test'."""
    h = int(hashlib.sha256(f"{seed}:{stem}".encode("utf-8")).hexdigest(), 16) % 10000
    train_cut = int(train_frac * 10000)
    val_cut = train_cut + int(val_frac * 10000)
    if h < train_cut:
        return "train"
    if h < val_cut:
        return "val"
    return "test"


def _episodes_in_item(item: Dict[str, Any]) -> List[str]:
    """Extract every source-episode stem an item depends on."""
    prov = item.get("provenance") or {}
    eps: List[str] = []
    if "episode" in prov and prov["episode"]:
        eps.append(str(prov["episode"]))
    if "episode_a" in prov and prov["episode_a"]:
        eps.append(str(prov["episode_a"]))
    if "episode_b" in prov and prov["episode_b"]:
        eps.append(str(prov["episode_b"]))
    if "episodes" in prov and isinstance(prov["episodes"], list):
        for e in prov["episodes"]:
            if isinstance(e, dict) and e.get("episode"):
                eps.append(str(e["episode"]))
    return eps


def _list_shards(api: HfApi, repo_id: str, dataset_folder: str, level: int) -> List[str]:
    prefix = f"{dataset_folder}/level_{level}/"
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as exc:
        logger.warning(f"L{level}: list_repo_files failed: {exc}")
        return []
    return sorted(
        f for f in files
        if f.startswith(prefix)
        and f.endswith(".jsonl")
        and SHARD_RE.search(f)  # only shard files, not train/val/test
    )


def _stream_items(api_repo_id: str, shard_paths: Iterable[str]) -> Iterable[Dict[str, Any]]:
    for path in shard_paths:
        try:
            local = hf_hub_download(repo_id=api_repo_id, filename=path, repo_type="dataset")
        except Exception as exc:
            logger.warning(f"download failed for {path}: {exc}")
            continue
        with open(local, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-folder", type=str, required=True,
                        help="e.g. factorynet_qa_150k")
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--keep-shards", action="store_true",
                        help="Don't delete the old shard files after writing splits.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and report counts but don't upload or delete.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not 0 < args.train_frac < 1 or not 0 < args.val_frac < 1:
        raise SystemExit("--train-frac and --val-frac must be in (0,1)")
    test_frac = 1.0 - args.train_frac - args.val_frac
    if test_frac <= 0:
        raise SystemExit("train+val must be < 1 (need room for test)")
    print(f"Split fractions: train={args.train_frac} val={args.val_frac} test={test_frac:.4f} (seed={args.seed})")

    api = HfApi(token=resolve_hf_token())

    # --- Phase 1: read all items, decide splits ----------------------------
    per_level_buckets: Dict[int, Dict[str, List[Dict[str, Any]]]] = {
        lv: {"train": [], "val": [], "test": []} for lv in args.levels
    }
    per_level_drops: Dict[int, int] = {lv: 0 for lv in args.levels}
    global_episode_split_counts: Counter = Counter()
    seen_episodes: Set[str] = set()

    for level in args.levels:
        shards = _list_shards(api, args.repo_id, args.dataset_folder, level)
        print(f"\n[L{level}] {len(shards)} shards:")
        for s in shards:
            print(f"  - {s}")
        if not shards:
            continue
        n_items = 0
        for item in _stream_items(args.repo_id, shards):
            n_items += 1
            eps = _episodes_in_item(item)
            if not eps:
                # No episode info — drop, can't assign.
                per_level_drops[level] += 1
                continue
            ep_splits = {
                _split_for_episode(e, args.seed, args.train_frac, args.val_frac)
                for e in eps
            }
            for e in eps:
                if e not in seen_episodes:
                    seen_episodes.add(e)
                    global_episode_split_counts[
                        _split_for_episode(e, args.seed, args.train_frac, args.val_frac)
                    ] += 1
            if len(ep_splits) > 1:
                # Cross-split paired/ranking — drop.
                per_level_drops[level] += 1
                continue
            split = next(iter(ep_splits))
            per_level_buckets[level][split].append(item)
        print(f"[L{level}] read {n_items} items")

    # --- Report ------------------------------------------------------------
    print("\n=== Episode-level split (global) ===")
    total_eps = sum(global_episode_split_counts.values())
    for sp in ("train", "val", "test"):
        c = global_episode_split_counts.get(sp, 0)
        pct = (100 * c / total_eps) if total_eps else 0
        print(f"  {sp}: {c} episodes ({pct:.1f}%)")

    print("\n=== Per-level Q&A counts after split ===")
    for level in args.levels:
        b = per_level_buckets[level]
        kept = len(b["train"]) + len(b["val"]) + len(b["test"])
        dropped = per_level_drops[level]
        total = kept + dropped
        if total == 0:
            print(f"  L{level}: no items found")
            continue
        print(
            f"  L{level}: kept={kept}, dropped={dropped} "
            f"({100*dropped/total:.1f}%) | "
            f"train={len(b['train'])} val={len(b['val'])} test={len(b['test'])}"
        )

    if args.dry_run:
        print("\n--dry-run: no writes / uploads / deletions.")
        return

    # --- Phase 2: write splits to a temp dir, upload, optionally delete shards ---
    print("\n=== Writing + uploading split files ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for level in args.levels:
            b = per_level_buckets[level]
            level_dir = tmp_root / f"level_{level}"
            level_dir.mkdir(parents=True, exist_ok=True)
            for split_name, items in b.items():
                if not items:
                    continue
                # Use HF-canonical name "validation" for val
                hf_name = "validation" if split_name == "val" else split_name
                out_path = level_dir / f"{hf_name}.jsonl"
                with out_path.open("w", encoding="utf-8") as fh:
                    for item in items:
                        fh.write(json.dumps(item, separators=(",", ":"), ensure_ascii=False) + "\n")
                remote_path = f"{args.dataset_folder}/level_{level}/{hf_name}.jsonl"
                print(f"  uploading L{level} {hf_name}: {len(items)} items -> {remote_path}")
                api.upload_file(
                    path_or_fileobj=str(out_path),
                    path_in_repo=remote_path,
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    commit_message=f"L{level} {hf_name} split ({len(items)} items, seed={args.seed})",
                )

    # --- Phase 3: delete old shards ---------------------------------------
    if not args.keep_shards:
        print("\n=== Deleting old shard files ===")
        for level in args.levels:
            shards = _list_shards(api, args.repo_id, args.dataset_folder, level)
            for shard_path in shards:
                try:
                    api.delete_file(
                        path_in_repo=shard_path,
                        repo_id=args.repo_id,
                        repo_type="dataset",
                        commit_message=f"Drop legacy shard {shard_path} (replaced by train/val/test)",
                    )
                    print(f"  deleted {shard_path}")
                except HfHubHTTPError as exc:
                    print(f"  delete failed for {shard_path}: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
