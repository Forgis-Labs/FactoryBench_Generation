"""Rescore the noise-substitution experiment with signed chance correction.

The paper's Appendix noise-substitution table currently reports clipped
chance-corrected means; we switch to the signed transform used everywhere
else in the paper. Same rubric, same replies, same questions — only the
final transform changes.

Reads:
  output/_archive/run_2026-04-28/replies/level*/<model>/*_answer.json    (Orig)
  output/_archive/run_2026-04-28/noised/replies_noised/...                (Sub)
  output/_archive/run_2026-04-28/prompts/level*/level*_NNNN.json          (prompt -> qa_pair_id)
  output/_archive/run_2026-04-28/questions/level*/<uuid>.json             (Orig questions)
  output/_archive/run_2026-04-28/noised/questions_noised/level*/<uuid>.json (Sub questions; same bounds by construction)

Writes: a JSON report and prints a paper-ready table with signed CC means
per (model, level, condition).

Free-form Level-4 items (Q&A with answer_format='free_form') keep their
cached llm_judge_score — no chance correction applies there.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.evaluation.chance_correct import chance_correct
from src.evaluation.run_foundry_eval import score_prediction

MODELS = ["claude-haiku-4-5", "DeepSeek-V3_1", "gpt-5_1-1", "Mistral-Large-3"]
MODEL_DISPLAY = {
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "DeepSeek-V3_1":    "DeepSeek V3.1",
    "gpt-5_1-1":        "gpt-5.1",
    "Mistral-Large-3":  "Mistral-Large-3",
}
LEVELS = ["level1", "level2", "level3", "level4"]


def _build_prompt_to_qid(prompts_dir: Path) -> Dict[str, str]:
    """{prompt_stem: qa_pair_id} for every prompt file."""
    out: Dict[str, str] = {}
    for pf in prompts_dir.glob("*.json"):
        try:
            p = json.loads(pf.read_text(encoding="utf-8"))
            qid = (p.get("metadata") or {}).get("qa_pair_id")
            if qid:
                out[pf.stem] = qid
        except Exception:
            continue
    return out


def _build_qid_to_question(questions_dir: Path) -> Dict[str, Dict[str, Any]]:
    """{qid: question dict}. Questions are stored under filename <uuid>.json."""
    out: Dict[str, Dict[str, Any]] = {}
    for qf in questions_dir.glob("*.json"):
        try:
            q = json.loads(qf.read_text(encoding="utf-8"))
            qid = q.get("id") or qf.stem
            out[qid] = q
        except Exception:
            continue
    return out


def _rescore_reply(rep: Dict[str, Any], q: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Return (raw, signed_cc). None if unscorable."""
    fmt = rep.get("answer_format") or q.get("answer_format") or ""
    if fmt == "free_form":
        # Free-form items are judge-scored; no chance-correction.
        s = rep.get("llm_judge_score")
        if s is None:
            return None
        return (float(s), float(s))
    # For deterministic formats, recompute the raw score under current rules
    # (in case bounds have been touched) then apply signed CC.
    try:
        raw, _ = score_prediction(
            answer_format=fmt,
            prediction=rep.get("answer") or "",
            ground_truth=q.get("answer"),
            acceptance_bounds=q.get("acceptance_bounds"),
            question_text=rep.get("prompt") or "",
            judge_model="",
        )
    except Exception:
        raw = rep.get("score")
    if raw is None:
        return None
    signed = chance_correct(raw, fmt, q, clip=False)
    if signed is None:
        return None
    return (float(raw), float(signed))


def _summarise(replies_root: Path, prompts_root: Path, questions_root: Path,
               label: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Return {(model, level): {raw, signed, n}}."""
    stats: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"raw": [], "signed": [], "n_missing_q": 0}
    )
    for level in LEVELS:
        prompt_to_qid = _build_prompt_to_qid(prompts_root / level)
        qid_to_q     = _build_qid_to_question(questions_root / level)
        for model in MODELS:
            model_dir = replies_root / level / model
            if not model_dir.is_dir(): continue
            for rp in model_dir.glob("*_answer.json"):
                try:
                    rep = json.loads(rp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                cid = rep.get("custom_id") or rp.stem[:-len("_answer")]
                # cid like 'level1_0000_0'; strip trailing _<idx> to get prompt stem
                pstem = re.sub(r"_\d+$", "", cid)
                qid = prompt_to_qid.get(pstem)
                if qid is None:
                    stats[(model, level)]["n_missing_q"] += 1
                    continue
                q = qid_to_q.get(qid)
                if q is None:
                    stats[(model, level)]["n_missing_q"] += 1
                    continue
                r = _rescore_reply(rep, q)
                if r is None: continue
                raw, signed = r
                stats[(model, level)]["raw"].append(raw)
                stats[(model, level)]["signed"].append(signed)
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (m, l), b in stats.items():
        if not b["raw"]:
            out[(m, l)] = {"raw": None, "signed": None, "n": 0, "n_missing_q": b["n_missing_q"]}
        else:
            out[(m, l)] = {
                "raw":    statistics.fmean(b["raw"])    * 100,
                "signed": statistics.fmean(b["signed"]) * 100,
                "n":      len(b["raw"]),
                "n_missing_q": b["n_missing_q"],
            }
    print(f"\n=== {label} ===")
    print(f"{'model':<20s} {'lvl':<7s} {'n':>4s}  {'raw%':>7s}  {'signed%':>8s}  missing_q")
    for (m, l), s in sorted(out.items()):
        raw = f"{s['raw']:7.2f}" if s['raw'] is not None else "     na"
        sig = f"{s['signed']:8.2f}" if s['signed'] is not None else "      na"
        print(f"{m:<20s} {l:<7s} {s['n']:>4d}  {raw}  {sig}  {s['n_missing_q']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-root", type=Path,
                    default=Path("output/_archive/run_2026-04-28"))
    ap.add_argument("--out", type=Path, default=Path("output/noise_sub_signed.json"))
    args = ap.parse_args()

    orig_stats = _summarise(
        replies_root   = args.archive_root / "replies",
        prompts_root   = args.archive_root / "prompts",
        questions_root = args.archive_root / "questions",
        label = "ORIGINAL",
    )
    sub_stats = _summarise(
        replies_root   = args.archive_root / "noised" / "replies_noised",
        prompts_root   = args.archive_root / "noised" / "prompts_noised",
        questions_root = args.archive_root / "noised" / "questions_noised",
        label = "NOISE-SUBSTITUTED",
    )

    # Paper-ready table
    print("\n\n=== Paper-ready table (signed CC%, L1-L3; raw judge %, L4) ===\n")
    header = f"{'Model':<20s} " + "  ".join([f"{lvl:<19s}" for lvl in LEVELS])
    print(header)
    print(f"{'':<20s} " + "  ".join(["Orig    Sub     Δ    " for _ in LEVELS]))
    for m in MODELS:
        row = f"{MODEL_DISPLAY[m]:<20s} "
        for l in LEVELS:
            o = orig_stats.get((m, l), {}).get("signed")
            s = sub_stats.get((m, l),  {}).get("signed")
            if o is None or s is None:
                row += f"{'na':<19s}  "
            else:
                row += f"{o:5.1f}  {s:5.1f}  {(s-o):+5.1f}  "
        print(row)

    args.out.write_text(json.dumps({
        "orig": {f"{m}|{l}": v for (m, l), v in orig_stats.items()},
        "sub":  {f"{m}|{l}": v for (m, l), v in sub_stats.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
