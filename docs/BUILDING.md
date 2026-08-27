# Building the FactoryBench papers

Every PDF in this repository is generated from one set of sources. Prose lives
in `paper/`; everything under `neurips_tex/`, `iclr_tex/` and `workshop_tex/`
is a wrapper whose only job is to load a venue style package and flip toggles.

**Never edit a wrapper to change content, and never copy content between venue
directories.** If two venues disagree about a number, the copy is the bug.

## Quick start

```bash
make -C docs neurips                # neurips_tex/main.pdf       (real authors)
make -C docs anon                   # neurips_tex/main_anon.pdf  (double-blind)
make -C docs iclr                   # iclr_tex/main_iclr.pdf     (ICLR 2027)
make -C docs workshop-wmphysai      # one target per venue
make -C docs archives               # dist/*.pdf + dist/*-source.zip per venue
make -C docs clean                  # drop aux files, keep the PDFs
```

`make` is not on `PATH` on a stock Windows + MiKTeX box. Use Git Bash with
`mingw32-make -C docs <target>` (Strawberry Perl ships one), or run the four
passes by hand:

```bash
cd docs/workshop_tex
pdflatex -interaction=nonstopmode main_wmphysai.tex && bibtex main_wmphysai \
  && pdflatex -interaction=nonstopmode main_wmphysai.tex \
  && pdflatex -interaction=nonstopmode main_wmphysai.tex
```

Engine is `pdflatex` + `bibtex`, four passes. Not latexmk, not tectonic.

> `-interaction=nonstopmode` means a build with missing figures or undefined
> citations still exits 0 and still writes a PDF. **Always read the log**, or
> use `make archives`, which fails loudly on exactly those conditions.

## Layout

```
docs/
  paper/                     ← the only place prose lives
    _preamble_common.tex       packages, palette, and the four venue toggles
    _titleauthor.tex           title + \ifanon-switched author block
    _content.tex               full-paper body: abstract .. conclusion
    _appendix.tex              shared appendix (~25pp), input by _content.tex
    _workshop_core.tex         trimmed ~6pp body, shared by ALL workshops
    _workshop_<venue>.tex      per-venue: title block, abstract, intro
    _workshop_TEMPLATE.tex     copy this to add a venue
    references.bib             ONE bibliography for every build

  neurips_tex/               NeurIPS wrappers + all figures + checklist
  iclr_tex/                  ICLR wrapper + ICLR style files
  workshop_tex/              workshop wrappers + neurips_2026.sty
  arxiv/                     FROZEN arXiv v1 record, see arxiv/README.md
  dist/                      build output (gitignored)
```

Figures live in `neurips_tex/figures/` only. The ICLR and workshop wrappers
point `\input@path` and `\graphicspath` there rather than keeping copies, which
is why those directories must stay siblings of `neurips_tex/`: move one a level
and the relative paths silently break.

## The four toggles

| Toggle | Declared in | Effect | Set by |
|---|---|---|---|
| `\ifanon` | each wrapper | blinded author block, `anonymous.4open.science` links | anon, iclr, all workshops: true |
| `\ifneuripschecklist` | `_preamble_common.tex` | include the NeurIPS paper checklist | NeurIPS builds only |
| `\paperbibstyle` | `_preamble_common.tex` | bibliography style | ICLR renews it to its own author-year `.bst` |
| `\ifworkshopcompact` | `_preamble_common.tex` | drop 3 optional floats, ~6pp body → ~4pp | 6-page-or-tighter workshops |

A wrapper sets these **after** `\input{../paper/_preamble_common}`. Declaring a
toggle again in the wrapper resets it.

## Adding a workshop

FactoryBench goes to several non-archival workshops, so the body is shared and
only framing differs. Three steps:

```bash
cp docs/paper/_workshop_TEMPLATE.tex   docs/paper/_workshop_<venue>.tex
cp docs/workshop_tex/main_TEMPLATE.tex docs/workshop_tex/main_<venue>.tex
# then add <venue> to WORKSHOPS in docs/Makefile
make -C docs archive-<venue>
```

In the **content** file edit three marked spots: the abstract, one or two
sentences in the third introduction paragraph connecting the four levels to the
venue's theme, and the two connector macros
(`\workshoprelatedworknote`, `\workshopclosingnote`) that
`_workshop_core.tex` expands at the end of Related Work and the Conclusion.

In the **wrapper** edit `\workshoptitle{}`, record the CFP terms in the comment
block beside it, set `\herowidth` (see below), and, if the limit is under 8
pages, `\workshopcompacttrue`.

`\herowidth` scales the pipeline figure uniformly. Shrinking it pulls the
abstract onto page 1, which is worth doing: 0.82 for `wmphysai` and 0.88 for
`robotlearning` both achieve it, and `wmphysai` gains a body page as a result.
Find the value by rebuilding and checking which page the Introduction starts
on.

Do not copy body prose. Related work, framework, generation, evaluation,
limitations and conclusion all come from `_workshop_core.tex`, so fixing a
result once corrects every workshop PDF on the next compile.

### Page budget

Each venue's limit comes from its CFP, and each wrapper records the CFP URL,
limit, blinding rule, OpenReview id and deadline beside its `\workshoptitle`.
As submitted (checked 2026-08-25):

| venue | CFP limit | main pages | shape |
|---|---|---|---|
| `wmphysai` | 8, refs excluded | 8 | full body, no appendix |
| `physunderstanding` | 9, refs **and appendix** excluded | 9 | full body + shared appendix |
| `robotlearning` | 6, refs excluded | 6 | `\workshopcompacttrue` |

All three sit exactly at their limit, so **any addition to
`_workshop_core.tex` pushes at least one of them over**. Re-check with
`make -C docs archives` and count to the page carrying the `References`
heading; if body text appears above it on that page, that page counts.

Compact mode drops the benchmark-comparison table, the levels-examples table
and the FactoryWave collage, and buys back ~2 pages. Nothing load-bearing is
gated: every claim and number survives, only illustrative float treatments are
cut. It also drops the 11 MB collage, taking the PDF from ~13 MB to ~0.4 MB.

### Cross-references

A workshop build has **no appendix**. Never `\ref` an `app:*` label from a
`_workshop_*.tex` file: the build will emit an undefined reference and
`nonstopmode` will not stop for it.

## Submission archives

`make -C docs archives` runs `make_archive.sh` per venue, which stages a flat
copy, rewrites the cross-directory paths to local ones, compiles it **inside
the staged directory** to prove the copy is complete, and only then zips it. It
aborts on any LaTeX error, undefined citation or reference, or missing input.

Each venue produces:

- `dist/factorybench-<venue>-workshop.pdf`, the PDF to upload
- `dist/factorybench-<venue>-workshop-source.zip`, flat sources that build with
  a bare `pdflatex main` (includes `main.bbl`, which portals that do not run
  bibtex require)

Only the figures a build actually reads are copied in. An archive carrying
unreferenced assets is not a clean one.

## Anonymity checklist

Before uploading a double-blind submission:

```bash
pdftotext dist/factorybench-<venue>-workshop.pdf - | \
  grep -inE 'forgis|kth|xelerit|github\.com/Forgis'
```

Page 1 must read `Anonymous Author(s)` and the code link must resolve to
`anonymous.4open.science`. Check the document info dictionary too
(`pdfinfo`): `Author` and `Title` must be empty.

The Hugging Face dataset URL is **not** a leak and is deliberately shown: the
org is named `FactoryBench`, which is the paper's own name and identifies no
author. Self-citations in the bibliography are expected and permitted, cited in
the third person like any other prior work; note ICLR's author-year style
renders them inline as "Petersen et al. (2026a)", which is more visible than a
bracketed number.

Note that `main.pdf` (the deanonymised NeurIPS build) currently still blinds its
author block, because `neurips_2026.sty`'s `eandd` option forces anonymity
unless `final` or `preprint` is also passed. Add `final` to its option list
before using it as a camera-ready.

## Recovered figures

Three appendix figures had no generator in this repository: `fig_judge_agreement`
needed an uncommitted reply tree, and `fig_probe_vs_readout` /
`fig_probe_layerwise` had no script anywhere. All three were recovered as vector
crops from the compiled workshop PDF that still contained them
(`docs/reference-drafts/main_physunderstanding.pdf`), using
`\includegraphics[page=N,trim=...,clip]` plus `pdfcrop`, and are now tracked
under `output/figures/`.

They are therefore **not regenerable from data**. If the underlying numbers ever
change, these three must be redrawn from scratch: the probing experiment's code
and outputs are gone.
