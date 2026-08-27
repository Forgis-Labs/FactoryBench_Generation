#!/usr/bin/env python
"""Build a single browsable page pairing each question with its feature board.

The boards are the point; this just puts the question text, the segmentation,
and the candidate boundary timestamps next to the figure so a solving session is
scroll-and-read rather than alt-tab. Boundary timestamps are listed as plain
text too, so they can be copied into the answer sheet without squinting at an
axis.

Reads only output/human_solve/pack/. Writes work/boards.html.

Usage:  python scripts/human_solve/make_index.py
"""
from __future__ import annotations

import csv
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stages import PACK, load, segment, series_files  # noqa: E402

WORK = Path(__file__).resolve().parents[2] / "output" / "human_solve" / "work"

CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;
     background:#fafafa;color:#1a1a1a}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;
       padding:10px 20px;z-index:10}
header a{margin-right:8px;font-size:12px;color:#2c6fbb;text-decoration:none}
section{max-width:1200px;margin:0 auto 40px;padding:0 20px}
h2{margin:28px 0 4px;font-size:17px}
.meta{color:#666;font-size:12px;margin-bottom:10px}
pre{background:#fff;border:1px solid #e2e2e2;border-radius:6px;padding:12px;
    white-space:pre-wrap;font:12.5px/1.5 ui-monospace,Consolas,monospace}
img{width:100%;border:1px solid #e2e2e2;border-radius:6px;background:#fff}
.edges{font:12.5px ui-monospace,Consolas,monospace;background:#fff7e6;
       border:1px solid #f0d9a8;border-radius:6px;padding:8px 12px;margin:8px 0}
.seq{color:#444}
@media(prefers-color-scheme:dark){
  body{background:#141414;color:#e8e8e8}header{background:#1c1c1c;border-color:#333}
  pre,img{background:#1c1c1c;border-color:#333}
  .edges{background:#2a2418;border-color:#5a4a24}
  .meta,.seq{color:#aaa}}
"""


def main():
    index = list(csv.DictReader((PACK / "index.csv").open(encoding="utf-8")))
    parts = [f"<style>{CSS}</style>", "<header>"]
    parts += [f'<a href="#i{r["n"]}">{r["n"]}</a>' for r in index]
    parts.append("</header><section>")

    for r in index:
        d = PACK / r["dir"]
        n = int(r["n"])
        q = html.escape((d / "question.txt").read_text(encoding="utf-8"))
        parts.append(f'<h2 id="i{n}">item {n:02d}</h2>')
        parts.append(f'<div class="meta">level {r["level"]} &middot; template '
                     f'{r["template_id"]} &middot; {r["template_type"]} &middot; '
                     f'{r["n_timesteps"]} steps &middot; {r["dir"]}</div>')
        parts.append(f"<pre>{q}</pre>")

        for src in series_files(d):
            s = load(d, src)
            segs = segment(s)
            seq = " &rarr; ".join(g["label"] for g in segs)
            edges = ", ".join(f'{g["label"]}@{g["t0"]:.0f}' for g in segs)
            suffix = "" if src.stem == "series" else f"_{src.stem.split('_')[-1]}"
            parts.append(f'<div class="edges"><b>{src.stem}</b> '
                         f'<span class="seq">{seq}</span><br>onsets [ms]: {edges}'
                         f'<br>window ends at {s["t"][-1]:.0f} ms, '
                         f'dt&asymp;{s["dt"]:.0f} ms</div>')
            parts.append(f'<img src="figs/board_{n:02d}{suffix}.png" loading="lazy">')

    parts.append("</section>")
    out = WORK / "boards.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out}  ({len(index)} items)")


if __name__ == "__main__":
    main()
