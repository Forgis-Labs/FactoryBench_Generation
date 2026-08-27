#!/usr/bin/env python
"""Export a *blinded* pack so a human can attempt the benchmark questions.

The sample file carries `answer`, `acceptance_bounds`, `root_cause` and a
`provenance` block inline, and several provenance fields hand over the gold
outright (`phase_start_in_subseries` for L1 T1, `fault_label` for L2,
`event_time_ms` for L3). Those go to gold.

No provenance reaches the pack at all. Naming the source dataset and episode
uuid would let a solver with the corpus in hand look the answer up instead of
solving, so the join keys (dataset, episode, subseries offset) go to _join.csv
on the scoring side. The item `id` is the only handle the pack carries, and it
round-trips to the source file once the answers are collected.

The provenance split fails closed: only keys in SAFE_PROV reach _join.csv,
anything unrecognised is treated as answer-bearing and diverted to _gold.json.
New generator fields therefore leak into gold, never into the join sheet.

  pack/<level>/<nn>_<id8>/question.txt   -- prompt + acronym legend only
  pack/<level>/<nn>_<id8>/series.csv     -- the time series, one row per step
  pack/<level>/<nn>_<id8>/plot.png       -- quicklook of every channel
  pack/<level>/<nn>_<id8>/meta.json      -- item id and shape, no provenance
  pack/index.csv                         -- one row per item, no provenance
  answers_template.csv                   -- blank sheet the solver fills in
  _join.csv                              -- id -> dataset/episode/offset
  _gold.json                             -- gold + bounds + held-back fields

Hand out pack/ and answers_template.csv. Keep _join.csv and _gold.json back
until the answers are in.

Usage:
  python scripts/rebuttal/export_human_solve_pack.py --level 1
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "output" / "sampled_qa_25perlevel.json"
OUT = REPO / "output" / "human_solve"

ROW_RE = re.compile(r"^t=(-?[\d.]+):\s*(.*)$")

# Top-level fields that are the answer, or trivially reveal it.
GOLD_FIELDS = ("answer", "acceptance_bounds", "root_cause")

# Provenance keys safe to put in _join.csv: they identify *where* the window
# came from, which is what you need to re-derive gold later, but say nothing
# about what the answer is. Everything else is diverted to _gold.json. Neither
# set goes anywhere near pack/.
SAFE_PROV = {
    "dataset", "episode", "subseries_start_index", "subseries_length",
    "sampled_subfolder", "counterpart_subfolder", "sampler",
    # paired-episode comparison items (L2/L3)
    "dataset_a", "machine_id_a", "episode_a", "subseries_start_a",
    "dataset_b", "machine_id_b", "episode_b", "subseries_start_b",
}
# Nested under provenance.relevance / relevance_a / relevance_b. `fault_id`,
# `target_phases` and `phase_overlap` name the thing being asked about, so only
# the sampling bookkeeping survives.
SAFE_RELEVANCE = {"sampler", "locality", "validated"}
RELEVANCE_KEYS = ("relevance", "relevance_a", "relevance_b")


def split_provenance(prov: dict | None) -> tuple[dict, dict]:
    """Return (safe join keys, held-back answer-bearing keys)."""
    safe: dict = {}
    held: dict = {}
    for k, v in (prov or {}).items():
        if k in RELEVANCE_KEYS and isinstance(v, dict):
            s = {kk: vv for kk, vv in v.items() if kk in SAFE_RELEVANCE}
            h = {kk: vv for kk, vv in v.items() if kk not in SAFE_RELEVANCE}
            if s:
                safe[k] = s
            if h:
                held[k] = h
        elif k in SAFE_PROV:
            safe[k] = v
        else:
            held[k] = v
    return safe, held


def collect_series(ctx: dict) -> list[tuple[str, dict, list[str], list[dict]]]:
    """Return [(suffix, acronym_mapping, columns, records), ...] for a context.

    Most items carry one series under `time_series`. The L1 T3 comparative
    template instead nests two under `series_a` / `series_b`, and reading only
    `time_series` silently exports an item with no data at all -- which is what
    happened to the comparative item in the first pack. Suffix is "" for the
    single case so existing filenames are unchanged.
    """
    if "time_series" in ctx:
        sides = [("", ctx)]
    else:
        sides = [(f"_{k.split('_')[-1]}", ctx[k])
                 for k in ("series_a", "series_b") if k in ctx]
    out = []
    for suffix, block in sides:
        mapping = (block.get("time_series_format") or {}).get("acronym_mapping", {})
        cols, recs = parse_series(block.get("time_series") or [])
        out.append((suffix, mapping, cols, recs))
    return out


def parse_series(rows: list[str]) -> tuple[list[str], list[dict]]:
    """Turn 't=0: fp0=1.2, fs0=0' strings into (columns, list-of-dicts)."""
    cols: list[str] = []
    out = []
    for r in rows:
        m = ROW_RE.match(r.strip())
        if not m:
            continue
        rec = {"t": float(m.group(1))}
        for kv in m.group(2).split(","):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            k = k.strip()
            if k not in cols:
                cols.append(k)
            try:
                rec[k] = float(v)
            except ValueError:
                rec[k] = v.strip()
        out.append(rec)
    return ["t"] + cols, out


def write_plot(cols, recs, mapping, path: Path, title: str):
    chans = [c for c in cols if c != "t"]
    # group channels by their acronym prefix so related signals share an axis
    groups: dict[str, list[str]] = {}
    for c in chans:
        pre = re.sub(r"\d+$", "", c)
        groups.setdefault(pre, []).append(c)
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.2 * n), sharex=True, squeeze=False)
    t = [r["t"] for r in recs]
    for ax, (pre, members) in zip(axes[:, 0], groups.items()):
        for c in members:
            ax.plot(t, [r.get(c) for r in recs], marker=".", ms=3, lw=1, label=c)
        full = mapping.get(members[0], pre)
        ax.set_ylabel(re.sub(r"_\d+$", "", full), fontsize=8)
        ax.legend(fontsize=6, ncol=6, loc="upper right")
        ax.grid(alpha=.3)
    axes[-1, 0].set_xlabel("t (ms)")
    # index ticks help for "which timestep" questions
    ax2 = axes[0, 0].twiny()
    ax2.set_xlim(axes[0, 0].get_xlim())
    ax2.set_xticks(t[:: max(1, len(t) // 12)])
    ax2.set_xticklabels([str(i) for i in range(0, len(t), max(1, len(t) // 12))],
                        fontsize=7)
    ax2.set_xlabel("step index", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=None, help="export only this level")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    items = json.loads(SRC.read_text(encoding="utf-8"))
    if args.level:
        items = [q for q in items if q["level"] == args.level]

    OUT.mkdir(parents=True, exist_ok=True)
    pack = OUT / "pack"
    gold, sheet, index, join = [], [], [], []
    dirs = {}

    for i, q in enumerate(items, 1):
        qid = q["id"]
        d = pack / f"level{q['level']}" / f"{i:02d}_{qid[:8]}"
        d.mkdir(parents=True, exist_ok=True)
        dirs[qid] = d

        ctx = q.get("context") or {}
        sides = collect_series(ctx)

        lines = [f"# item {i:02d}  (level {q['level']}, {q['template_type']})", ""]
        lines += [q["question"], ""]
        if q.get("options"):
            lines.append("Options:")
            for k, v in q["options"].items():
                lines.append(f"  {k}) {v}")
            lines.append("")
        legend: dict[str, str] = {}
        for suffix, mapping, cols, recs in sides:
            label = f"Series{suffix.upper().replace('_', ' ')}"
            lines.append(f"{label}: {len(recs)} timesteps, {len(cols) - 1} channels "
                         f"(see series{suffix}.csv / plot{suffix}.png)")
            legend.update(mapping)
        if legend:
            lines.append("")
            lines.append("Channel legend:")
            for k, v in legend.items():
                lines.append(f"  {k} = {v}")
        (d / "question.txt").write_text("\n".join(lines), encoding="utf-8")

        for suffix, mapping, cols, recs in sides:
            if not recs:
                continue
            with open(d / f"series{suffix}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["step"] + cols)
                w.writeheader()
                for j, r in enumerate(recs):
                    w.writerow({"step": j, **r})
            if not args.no_plots:
                write_plot(cols, recs, mapping, d / f"plot{suffix}.png",
                           f"item {i:02d} - L{q['level']} T{q['template_id']}"
                           f"{suffix and ' (' + suffix[1:] + ')'}")
        # downstream tooling keys off the first (or only) series
        cols, recs = (sides[0][2], sides[0][3]) if sides else ([], [])

        safe_prov, held_prov = split_provenance(q.get("provenance"))
        # pack side: shape of the item, and the id that joins it back later
        meta = {"n": i, "id": qid, "level": q["level"],
                "template_id": q["template_id"],
                "template_type": q["template_type"],
                "hides": q.get("hides", []),
                "n_timesteps": len(recs),
                "n_channels": max(0, len(cols) - 1),
                "series": [{"name": f"series{s or ''}",
                            "n_timesteps": len(rr),
                            "n_channels": max(0, len(cc) - 1)}
                           for s, _m, cc, rr in sides]}
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        index.append({"n": i, "id": qid, "level": q["level"],
                      "template_id": q["template_id"],
                      "template_type": q["template_type"],
                      "n_timesteps": len(recs),
                      "dir": str((d.relative_to(pack)).as_posix())})
        # scoring side: where the window came from
        join.append({"n": i, "id": qid, "level": q["level"],
                     "template_id": q["template_id"],
                     "source_file": SRC.name,
                     "dataset": safe_prov.get("dataset", ""),
                     "episode": safe_prov.get("episode", ""),
                     "subseries_start_index": safe_prov.get("subseries_start_index", ""),
                     "subseries_length": safe_prov.get("subseries_length", ""),
                     "provenance_extra": json.dumps(
                         {k: v for k, v in safe_prov.items()
                          if k not in ("dataset", "episode",
                                       "subseries_start_index",
                                       "subseries_length")},
                         ensure_ascii=False),
                     "dir": str((d.relative_to(pack)).as_posix())})

        g = {"n": i, "id": qid, "level": q["level"],
             "template_id": q["template_id"]}
        for f in GOLD_FIELDS:
            if f in q:
                g[f] = q[f]
        g["held_back_provenance"] = held_prov
        gold.append(g)
        sheet.append({"n": i, "id": qid, "level": q["level"],
                      "template_id": q["template_id"],
                      "human_answer": "", "minutes": "", "tools_used": "",
                      "confidence_1_5": "", "notes": ""})

    (OUT / "_gold.json").write_text(json.dumps(gold, indent=2), encoding="utf-8")
    with open(pack / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    with open(OUT / "answers_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0].keys()))
        w.writeheader()
        w.writerows(sheet)
    with open(OUT / "_join.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(join[0].keys()))
        w.writeheader()
        w.writerows(join)

    audit(items, pack, dirs)
    print(f"{len(items)} items -> {pack}")
    print(f"hand out: {pack} and {OUT / 'answers_template.csv'}")
    print(f"hold back until answers are in: {OUT / '_join.csv'}, "
          f"{OUT / '_gold.json'}")


HELD_KEYS = ("phase_start_in_subseries", "phase_length", "fault_label",
             "event_time_ms", "event_index_alt", "prediction_index",
             "target_phases", "phase_overlap", "fault_id")


def audit(items, pack: Path, dirs: dict[str, Path]):
    """Check each item's own files for gold or join keys that escaped the split.

    Scoped per item, because a gold token showing up in a *different* item's
    prompt is coincidence, not a leak. Only distinctive gold strings are worth
    grepping: short common words like "normal" collide with ordinary prompt
    text ("abnormal force buildup"), and bare numeric golds collide with the
    series itself. Those cases rest on the fail-closed provenance split
    instead, which is the actual guarantee here.

    The join keys get their own check: an episode uuid or dataset name in the
    pack turns the item into a lookup for anyone holding the corpus, so they
    are treated as leaks even though they are not the answer.
    """
    hits = []
    index_txt = (pack / "index.csv").read_text(encoding="utf-8")
    for q in items:
        d = dirs[q["id"]]
        own = "\n".join(p.read_text(encoding="utf-8") for p in d.iterdir()
                        if p.suffix in (".txt", ".json", ".csv"))
        safe_prov, _ = split_provenance(q.get("provenance"))
        for k, v in safe_prov.items():
            if not isinstance(v, str) or len(v) < 8:
                continue
            if v in own:
                hits.append((q["id"], f"join:{k}", v))
            if v in index_txt:
                hits.append((q["id"], f"join:{k}", f"{v} (index.csv)"))
        # the prompt legitimately contains option text, so compare against
        # what we added on top of question.txt
        prompt = (d / "question.txt").read_text(encoding="utf-8")
        extra = own.replace(prompt, "")
        for f in GOLD_FIELDS:
            v = q.get(f)
            if not isinstance(v, str):
                continue
            distinctive = "_" in v or len(v) >= 8
            if distinctive and v in extra:
                hits.append((q["id"], f, v))
        for k in HELD_KEYS:
            # `extra`, not `own`: a couple of L2 distractors literally say
            # "(fault_id 11)" in their option text. That is the released
            # prompt's business, not something this export added.
            if k in extra:
                hits.append((q["id"], "provenance", k))
    if hits:
        print(f"LEAK: {len(hits)} gold value(s) found in pack/")
        for h in hits[:10]:
            print("  ", h)
    else:
        print(f"audit: {len(items)} items clean "
              "(no held-back provenance keys, no distinctive gold strings)")


if __name__ == "__main__":
    main()
