"""Pull `Forgis/FactoryBench:factorybench_qa/level_<N>/test.jsonl`, filter to
KUKA-grounded items only, and stage them as per-item JSON under
`output/kuka_qa/level<N>/` so the existing evaluation pipeline can consume
them without any per-source special-casing.

KUKA items are identified by ``provenance.dataset == "factorywave_kuka"``.
This is exactly the difference between the paper's Fig 3 (~1209/3132/297/1113
UR3-only) and Fig 4 (~1309/3487/321/1232 full test split) that the reviewer
flagged in W.2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(usecwd=True))

from huggingface_hub import HfApi, hf_hub_download


REPO = "Forgis/FactoryBench"
QA_ROOT = "factorybench_qa"
KUKA_DATASET = "factorywave_kuka"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("output/kuka_qa"))
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()

    token = os.getenv("HF_WRITE_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN")
    api = HfApi(token=token)

    totals: dict[int, tuple[int, int]] = {}
    for lvl in args.levels:
        remote = f"{QA_ROOT}/level_{lvl}/test.jsonl"
        local = hf_hub_download(args.repo, remote, repo_type="dataset", token=token)
        out_dir = args.out_root / f"level{lvl}"
        out_dir.mkdir(parents=True, exist_ok=True)
        n_all = n_kuka = 0
        with open(local, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                q = json.loads(line)
                n_all += 1
                prov = q.get("provenance") or {}
                if prov.get("dataset") != KUKA_DATASET:
                    continue
                qid = q.get("id") or f"level{lvl}_{n_kuka:05d}"
                # `run_foundry_eval` looks up questions by stem, so we name the file
                # after the question id to keep the round-trip clean.
                with open(out_dir / f"{qid}.json", "w", encoding="utf-8") as w:
                    json.dump(q, w, indent=2)
                n_kuka += 1
        totals[lvl] = (n_kuka, n_all)
        print(f"L{lvl}: {n_kuka}/{n_all} items are KUKA-grounded -> {out_dir}")

    grand_kuka = sum(k for k, _ in totals.values())
    grand_all = sum(a for _, a in totals.values())
    print(f"total: {grand_kuka}/{grand_all} released test-split items are KUKA-grounded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
