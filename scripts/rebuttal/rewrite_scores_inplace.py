"""Rewrite each reply's cached `score` field under the current acceptance_bounds.

Needed after `patch_l1_t7_margins.py`: the reply JSONs still carry stale
scores computed under the old (buggy) template-7 rule. The heatmap
generator and the paper table read `reply.score` directly, so unless we
rewrite the field they'll keep showing the pre-patch numbers.

Only touches numerical items (the branch whose scoring rule changed).
Preserves the original in `_score_pre_patch` so the change is auditable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.evaluation.run_foundry_eval import score_prediction


def _load_question(qroot: Path, level: str, custom_id: str):
    parts = custom_id.split("_")
    stripped = "_".join(parts[:-1]) if len(parts) >= 2 and parts[-1].isdigit() else custom_id
    stems = [stripped, custom_id,
             stripped[len(level) + 1:] if stripped.startswith(f"{level}_") else f"{level}_{stripped}",
             custom_id[len(level) + 1:] if custom_id.startswith(f"{level}_") else f"{level}_{custom_id}"]
    for s in stems:
        p = qroot / level / f"{s}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies-root", type=Path, required=True)
    ap.add_argument("--questions-root", type=Path, required=True)
    ap.add_argument("--levels", nargs="*", default=["level1"],
                    help="which level dirs to update (default: level1)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed = unchanged = skipped = missing_q = 0
    for level in args.levels:
        level_dir = args.replies_root / level
        if not level_dir.is_dir():
            print(f"missing: {level_dir}"); continue
        for model_dir in sorted(level_dir.iterdir()):
            if not model_dir.is_dir(): continue
            for rp in model_dir.glob("*_answer.json"):
                try:
                    rep = json.loads(rp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                fmt = rep.get("answer_format")
                if fmt != "numerical":
                    skipped += 1; continue
                cid = rep.get("custom_id") or rp.stem[:-len("_answer")]
                q = _load_question(args.questions_root, level, cid)
                if q is None:
                    missing_q += 1; continue
                try:
                    new_score, _ = score_prediction(
                        answer_format="numerical",
                        prediction=rep.get("answer") or "",
                        ground_truth=q.get("answer"),
                        acceptance_bounds=q.get("acceptance_bounds"),
                        question_text=rep.get("prompt") or "",
                        judge_model="",
                    )
                except Exception:
                    new_score = None
                if new_score is None:
                    skipped += 1; continue
                old_score = rep.get("score")
                if old_score is not None and abs(float(old_score) - float(new_score)) < 1e-6:
                    unchanged += 1; continue
                changed += 1
                if not args.dry_run:
                    if "_score_pre_patch" not in rep:
                        rep["_score_pre_patch"] = old_score
                    rep["score"] = float(new_score)
                    rp.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"changed: {changed}  unchanged: {unchanged}  skipped (non-numerical): {skipped}  missing_q: {missing_q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
