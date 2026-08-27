"""Re-draw multiple-choice options whose evidence the item never shows.

The four multi-select templates (L2.2, L2.3, L3.2, L3.3) chose their options
without checking whether the item's context carries the channels each statement
is about. On the released benchmark 46-74% of options were unsupported, and
2,747 items contained at least one. An item could ask whether "robot current
remains approximately constant" while showing no current channel at all.

Adding the missing channels is not an option: measured over the affected items
the median one would need 26 further channels and 24 of those do not exist in
its source dataset. So the option pool has to shrink instead, which is what
``utils.mc_availability`` now does at generation time. This script applies the
same correction to what is already published.

Repair, not regeneration. Each item keeps its id, its question, its context and
its window; only the option set and the answer string change. Wholesale
regeneration would mint new ids and shift per-level counts, which would break
the alignment between the figures and the release that the paper depends on.

For every affected item:

  1. reload the episode named in provenance and rebuild the same window,
  2. restrict the option pool to statements the *context* can support (the
     model has to answer from what it is shown, not from the full episode),
  3. re-draw four options and recompute each truth value against the episode
     rows with ``evaluate_mc_statement``, which is the same path generation
     uses, so the new ground truth is derived exactly as the original was.

Ground truth necessarily changes for these items. Any score previously
reported on them no longer applies.

Usage:
    python scripts/repair_unsupported_mc_options.py --episodes <dir> --workdir <dir>
    python scripts/repair_unsupported_mc_options.py --episodes <dir> --workdir <dir> --push
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

from src.question_generation.utils.mc_availability import (
    expand_feature,
    filter_lookup_by_availability,
)

load_dotenv(Path.cwd() / ".env")
load_dotenv()

REPO = "FactoryBench/FactoryBench"
SPLITS = ("train", "validation", "test")
TARGETS = {(2, 2), (2, 3), (3, 2), (3, 3)}
CATALOGUE_PATH = Path("data/mc_options/mc_options.json")


# --------------------------------------------------------------------------
# catalogue matching: released statements have per-item thresholds substituted,
# so exact string lookup fails. Match on the wording with numbers stripped.
# --------------------------------------------------------------------------
def _norm(text: str) -> str:
    t = re.sub(r"\([^)]*\)", " ", str(text).lower())
    t = re.sub(r"[-+]?\d*\.?\d+\s*%?", " ", t)
    return " ".join(re.sub(r"[^a-z ]", " ", t).split())


class Catalogue:
    def __init__(self, path: Path):
        self.records = json.loads(path.read_text(encoding="utf-8"))
        self._normed = [(_norm(o["statement"]).split(), o) for o in self.records]
        self.lookup = {o["id"]: o["statement"] for o in self.records}

    def match(self, text: str):
        words = _norm(text).split()
        best, score = None, 0
        for cand, option in self._normed:
            k = 0
            while k < min(len(words), len(cand)) and words[k] == cand[k]:
                k += 1
            if k > score:
                best, score = option, k
        return best if score >= 4 else None


def context_channels(ctx) -> set:
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except Exception:
            return set()
    if not isinstance(ctx, dict):
        return set()
    out = set()
    for holder in [ctx] + [v for v in ctx.values() if isinstance(v, dict)]:
        fmt = holder.get("time_series_format")
        if isinstance(fmt, dict) and isinstance(fmt.get("acronym_mapping"), dict):
            out |= set(fmt["acronym_mapping"].values())
        notes = holder.get("notes")
        if isinstance(notes, dict) and isinstance(notes.get("constant_features"), dict):
            out |= set(notes["constant_features"].keys())
    return out


def option_supported(option, channels) -> bool:
    groups = []
    for spec in (option.get("required_features") or []):
        expanded = expand_feature(spec)
        if expanded is None:
            return False
        groups.append(expanded)
    return True if not groups else any(g & channels for g in groups)


def item_is_affected(item, cat: Catalogue) -> bool:
    channels = context_channels(item.get("context"))
    opts = item.get("options") or {}
    if isinstance(opts, str):
        try:
            opts = json.loads(opts)
        except Exception:
            return False
    for text in opts.values():
        option = cat.match(text)
        if option and not option_supported(option, channels):
            return True
    return False


# --------------------------------------------------------------------------
# episode access
# --------------------------------------------------------------------------
class Episodes:
    """Normalized episode rows and metadata, looked up by episode id."""

    def __init__(self, roots):
        self.index, self.meta_index = {}, {}
        for root in roots:
            for path in Path(root).rglob("*.json"):
                name = path.name
                if name.endswith("_metadata.json"):
                    self.meta_index.setdefault(name[: -len("_metadata.json")], path)
                else:
                    self.index.setdefault(path.stem, path)
        self._cache = {}

    def rows(self, episode_id, split="flat"):
        """Rows for an episode.

        Counterfactual episodes are stored as one file holding both the
        baseline and the intervened run, keyed by split, so L3 has to ask for
        the half it wants. Flat (single-run) files ignore the split.
        """
        key = f"{episode_id}::{split}"
        if key in self._cache:
            return self._cache[key]
        path = self.index.get(episode_id)
        rows = None
        if path:
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    rows = loaded
                elif isinstance(loaded, dict):
                    # Combined counterfactual files hold both runs under
                    # separate keys. Prefer the requested split, then the
                    # single-run key, then whichever key holds rows at all:
                    # some L2 episodes live in this format even though the
                    # item was built from the run as a flat series.
                    rows = loaded.get(split) or loaded.get("flat") or loaded.get("baseline")
                    if not rows:
                        rows = next((v for v in loaded.values()
                                     if isinstance(v, list) and v), None)
            except Exception:
                rows = None
        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[key] = rows
        return rows

    def metadata(self, episode_id):
        path = self.meta_index.get(episode_id)
        if not path:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None


def rebuild_window(item, eps):
    """(context rows, post-event rows) for this item, reconstructed as the
    generator built them.

    The two levels differ and conflating them silently produces wrong ground
    truth. At L2 the window ends just before the event, so the post-event rows
    are simply the remainder of the same episode. At L3 the context is the
    baseline run while the outcome is the intervened run, the two live in one
    file under separate keys, and the window straddles the onset, so the
    post-event rows start at ``event_index_alt`` of the counterfactual half
    rather than after the window.
    """
    prov = item.get("provenance")
    if isinstance(prov, str):
        try:
            prov = json.loads(prov)
        except Exception:
            prov = {}
    prov = prov or {}
    start = prov.get("subseries_start_index")
    length = prov.get("subseries_length")
    if start is None or length is None:
        return None, None
    start, length = int(start), int(length)

    if item.get("level") == 3:
        base_id = str(prov.get("sampled_subfolder") or prov.get("episode") or "")
        alt_id = str(prov.get("counterpart_subfolder") or prov.get("episode") or "")
        onset = prov.get("event_index_alt")
        base = eps.rows(base_id, "baseline")
        alt = eps.rows(alt_id, "counterfactual")
        if not base or not alt or onset is None:
            return None, None
        onset = int(onset)
        if start < 0 or length <= 0 or start + length > len(base):
            return None, None
        if not (0 <= onset < len(alt)):
            return None, None
        return base[start: start + length], alt[onset:]

    rows = eps.rows(str(prov.get("episode") or ""))
    if not rows:
        return None, None
    if start < 0 or length <= 0 or start + length >= len(rows):
        return None, None
    return rows[start: start + length], rows[start + length:]


def reconstruction_matches(item, subseries) -> bool:
    """Does the rebuilt window actually hold the rows the item displays?

    Index-based reconstruction is only valid if the episode on disk is the one
    the item was generated from. Combined counterfactual files store two runs,
    and picking the wrong half yields plausible-looking rows that are not the
    ones shown. Comparing the first displayed row against the rebuilt window
    catches that; without this guard 37 items were rebuilt from the wrong run
    and would have shipped ground truth computed on rows the reader never sees.
    """
    ctx = item.get("context")
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except Exception:
            return False
    fmt = (ctx or {}).get("time_series_format") or {}
    mapping = fmt.get("acronym_mapping") or {}
    series = (ctx or {}).get("time_series") or []
    if not mapping or not series or not subseries:
        return False
    body = str(series[0]).partition(": ")[2]
    row = subseries[0]
    checked = 0
    for token in body.split(", "):
        acr, _, shown = token.partition("=")
        chan = mapping.get(acr)
        if chan is None or chan not in row or row[chan] is None:
            continue
        checked += 1
        try:
            if abs(float(shown) - float(row[chan])) > 0.011:
                return False
        except (TypeError, ValueError):
            if str(shown) != str(row[chan]):
                return False
    return checked > 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=Path, nargs="+", required=True,
                    help="Directories of normalized episode JSON (10 Hz).")
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--token-env", default="HF_WRITE_TOKEN")
    args = ap.parse_args()

    token = os.getenv(args.token_env)
    if not token:
        raise SystemExit(f"{args.token_env} not set")
    api = HfApi(token=token)
    args.workdir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    cat = Catalogue(CATALOGUE_PATH)
    eps = Episodes(args.episodes)
    print(f"episodes indexed: {len(eps.index)}  (metadata: {len(eps.meta_index)})")

    # generator-side builders, imported lazily so a missing dep is obvious
    from src.question_generation.level2.level2 import (
        build_multiselect_options_and_answer as build_l2)
    from src.question_generation.level3.level3 import (
        build_multiselect_options_and_answer as build_l3)
    templates = {}
    for lvl in (2, 3):
        path = Path(f"src/question_generation/level{lvl}/question_template.json")
        for t in json.loads(path.read_text(encoding="utf-8")):
            templates[(lvl, t["id"])] = t

    tally = collections.Counter()
    staged = []
    for lvl in (2, 3):
        builder = build_l2 if lvl == 2 else build_l3
        for split in SPLITS:
            remote = f"factorybench_qa/level_{lvl}/{split}.jsonl"
            local = Path(hf_hub_download(REPO, remote, repo_type="dataset",
                                         token=token, local_dir=args.workdir / "download"))
            out_rows = []
            for line in local.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                key = (item.get("level"), item.get("template_id"))
                if key in TARGETS and item_is_affected(item, cat):
                    tally["affected"] += 1
                    fixed = repair(item, cat, eps, templates.get(key), builder)
                    if fixed is None:
                        tally["could_not_repair"] += 1
                    else:
                        item = fixed
                        tally["repaired"] += 1
                out_rows.append(item)
            out = args.workdir / "fixed" / remote
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as fh:
                for r in out_rows:
                    fh.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n")
            staged.append((out, remote))
            print(f"  L{lvl}/{split}: {len(out_rows)} rows")

    print(f"\naffected={tally['affected']} repaired={tally['repaired']} "
          f"could_not_repair={tally['could_not_repair']}")

    if not args.push:
        print("dry run, nothing pushed. Re-run with --push.")
        return 0
    for path, remote in staged:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=remote,
                        repo_id=REPO, repo_type="dataset",
                        commit_message="L2.2/L2.3/L3.2/L3.3: re-draw options the context cannot support")
        print(f"pushed {remote}")
    return 0


def repair(item, cat: Catalogue, eps: Episodes, template, builder):
    """Return the item with a supported option set and recomputed answer."""
    if template is None:
        return None
    prov = item.get("provenance")
    if isinstance(prov, str):
        try:
            prov = json.loads(prov)
        except Exception:
            prov = {}
    prov = prov or {}
    episode_id = str(prov.get("episode") or "")
    subseries, post = rebuild_window(item, eps)
    if not subseries or not post:
        return None
    if not reconstruction_matches(item, subseries):
        return None

    channels = context_channels(item.get("context"))
    pool = filter_lookup_by_availability(
        cat.lookup, cat.records, [{c: 0 for c in channels}],
        level=item.get("level"), minimum=4,
    )
    if len(pool) < 4:
        return None

    kwargs = dict(
        answer_format=template.get("answer_format") or {},
        post_event_rows=post,
        mc_option_lookup=pool,
        episode_metadata=eps.metadata(episode_id),
    )
    # the two levels name the baseline argument differently
    try:
        options, answer = builder(baseline_subseries=subseries, **kwargs)
    except TypeError:
        options, answer = builder(subseries=subseries, **kwargs)

    if not options or not answer or len(answer) != len(options):
        return None
    # never ship an option the context still cannot support
    for text in options.values():
        opt = cat.match(text)
        if opt and not option_supported(opt, channels):
            return None

    item["options"] = options
    item["answer"] = answer
    return item


if __name__ == "__main__":
    raise SystemExit(main())
