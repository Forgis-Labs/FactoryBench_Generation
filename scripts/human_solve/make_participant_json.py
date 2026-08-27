#!/usr/bin/env python
"""Strip the sampled QA file down to what a human participant may see.

`output/sampled_qa_25perlevel.json` cannot be handed out as-is. Three of its
fields are answers and three more are hints:

  answer, acceptance_bounds   the key
  provenance                  carries phase_start_in_subseries, fault_label,
                              event_time_ms, prediction_index and relevance,
                              which are the answers to their respective
                              templates in plain text
  level, template_id          difficulty and question family
  template_type               the worst of the hints: 'predictive' tells the
                              solver the asked-for event lies past the end of
                              the released window, which is most of the work on
                              those items
  hides                       says which entities were redacted, so it tells
                              the solver what kind of thing to name
  root_cause                  present on the 25 level-4 troubleshooting items
                              and is the free-text answer to them

What survives is the prompt, its options, the series the prompt refers to, and
`id`. `id` is the lookup key: scoring joins the returned answers back onto the
original file on it, so nothing else needs to travel with the participant copy.

Order is shuffled under a fixed seed, because the source file is sorted by
level and position alone would otherwise give the difficulty away.

Usage:
  python scripts/human_solve/make_participant_json.py
  python scripts/human_solve/make_participant_json.py --out somewhere.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "output" / "sampled_qa_25perlevel.json"
DST = REPO / "output" / "human_baseline" / "participant_samples.json"

KEEP = ("id", "question", "options", "context")
DROP = ("level", "template_id", "template_type", "hides",
        "answer", "acceptance_bounds", "provenance", "root_cause")
SEED = 20260728


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=DST)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    src = json.loads(args.src.read_text(encoding="utf-8"))

    unexpected = {k for r in src for k in r} - set(KEEP) - set(DROP)
    if unexpected:
        raise SystemExit(
            f"unrecognised field(s) {sorted(unexpected)} in {args.src.name}. "
            "Classify them as KEEP or DROP before sharing anything.")

    out = [{k: r[k] for k in KEEP if k in r} for r in src]
    random.Random(args.seed).shuffle(out)

    # a stripped record must carry nothing but the four allowed keys, and no
    # nested key may look like a label
    banned = ("answer", "phase_start", "phase_name", "phase_length", "fault",
              "event_time", "event_index", "prediction_index", "relevance",
              "acceptance", "provenance", "level", "template", "hides",
              "episode", "dataset", "machine_id", "subseries_start",
              "root_cause", "label", "gold", "solution")

    def scan(node, trail=""):
        if isinstance(node, dict):
            for k, v in node.items():
                low = str(k).lower()
                if any(b in low for b in banned):
                    raise SystemExit(f"leak: key '{trail}/{k}' survived stripping")
                scan(v, f"{trail}/{k}")
        elif isinstance(node, list):
            for v in node:
                scan(v, trail)

    for r in out:
        assert set(r) <= set(KEEP), sorted(set(r) - set(KEEP))
        scan(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{len(out)} items -> {args.out}")
    print(f"kept:    {', '.join(KEEP)}")
    print(f"dropped: {', '.join(DROP)}")
    print(f"shuffled with seed {args.seed}; join results back on `id`")
    print(f"size: {args.src.stat().st_size/1e6:.2f} MB -> "
          f"{args.out.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
