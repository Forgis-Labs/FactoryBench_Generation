"""Count characters per assistant response in a Claude Code session transcript.

Reads the newline-delimited JSON transcript produced by Claude Code
(`~/.claude/projects/<slug>/<session-id>.jsonl`), extracts each assistant
turn's text output (concatenating all text blocks in the turn, ignoring
thinking blocks and tool_use blocks), and prints per-turn character
counts plus summary statistics.

Usage:
    python scripts/count_assistant_chars.py                      # newest .jsonl for this project
    python scripts/count_assistant_chars.py <session_id_or_path> # specific session
    python scripts/count_assistant_chars.py -n 20                # show only last 20 turns
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PROJECT_DIR = Path(
    "C:/Users/ymerz/.claude/projects/"
    "c--Users-ymerz-OneDrive-Documents-Work-Forgis-FactoryBench-Generation"
)


def _newest_transcript(project_dir: Path) -> Path:
    files = sorted(project_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no .jsonl transcripts under {project_dir}")
    return files[-1]


def _extract_text(msg: Dict[str, Any]) -> str:
    """Concatenate every user-facing text block from one assistant message.

    Skips thinking blocks (not shown to the user), tool_use blocks (invocations),
    and tool_result blocks (from other roles). Only 'text' blocks count.
    """
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict): continue
        btype = block.get("type")
        if btype == "text":
            t = block.get("text")
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts)


def load_assistant_turns(path: Path) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw: continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "assistant": continue
            msg = rec.get("message") or {}
            if msg.get("role") != "assistant": continue
            text = _extract_text(msg)
            if not text.strip(): continue  # tool-only turn, no user-facing text
            turns.append({
                "line":      line_no,
                "uuid":      rec.get("uuid"),
                "timestamp": rec.get("timestamp"),
                "chars":     len(text),
                "words":     len(text.split()),
                "preview":   text.strip()[:80].replace("\n", " "),
            })
    return turns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", default=None,
                    help="session id or path to .jsonl (default: newest in project dir)")
    ap.add_argument("-n", "--last", type=int, default=None,
                    help="show only the last N turns in the per-turn table")
    ap.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    args = ap.parse_args()

    if args.session is None:
        path = _newest_transcript(args.project_dir)
    else:
        p = Path(args.session)
        if p.exists():
            path = p
        else:
            path = args.project_dir / f"{args.session}.jsonl"
            if not path.exists():
                raise SystemExit(f"transcript not found: {path}")

    print(f"transcript: {path}")
    turns = load_assistant_turns(path)
    if not turns:
        print("no assistant turns with user-facing text found")
        return 0

    chars = [t["chars"] for t in turns]
    words = [t["words"] for t in turns]

    print()
    print(f"n_turns={len(turns)}")
    print(f"chars: min={min(chars)}  median={int(statistics.median(chars))}  "
          f"mean={int(statistics.fmean(chars))}  p90={int(sorted(chars)[int(0.9*len(chars))-1])}  max={max(chars)}  total={sum(chars):,}")
    print(f"words: min={min(words)}  median={int(statistics.median(words))}  "
          f"mean={int(statistics.fmean(words))}  max={max(words)}")

    to_show = turns if args.last is None else turns[-args.last:]
    print()
    print(f"{'#':>4s}  {'chars':>6s}  {'words':>6s}  timestamp             preview")
    for i, t in enumerate(to_show, start=(len(turns) - len(to_show) + 1)):
        ts = (t.get("timestamp") or "")[:19].replace("T", " ")
        print(f"{i:>4d}  {t['chars']:>6d}  {t['words']:>6d}  {ts:<20s}  {t['preview']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
