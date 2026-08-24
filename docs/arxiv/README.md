# FROZEN — arXiv v1 submission record

This directory is the source bundle exactly as submitted to arXiv on
2026-05-25, kept as a record of what was published. **Do not edit it.**

It predates the shared-source refactor, so its `main.tex` is a standalone
~1500-line copy of prose that now lives in `../paper/_content.tex` and
`../paper/_appendix.tex`, and it carries its own `references.bib` and
`neurips_2026.sty`. It has already diverged from the current paper: among other
things it still names the dataset `Forgis/FactoryBench` (now
`FactoryBench/FactoryBench`), reports Level 4 as multiple choice (reverted to
free-form in commit `39e4075`), and describes a train/validation/test partition
the release no longer has.

**A fix applied here reaches nothing else, and a fix applied elsewhere never
reaches here.** Treat any disagreement between this directory and `../paper/`
as this directory being out of date.

To prepare a v2, do not edit these files. Build from the shared sources
instead:

```bash
make -C .. neurips        # or anon / iclr
```

and, if arXiv specifically needs a preprint banner rather than a submission
one, pass `preprint` instead of `eandd` in the wrapper's
`\usepackage{neurips_2026}` option list.

The dated duplicate of this bundle (`docs/arxiv_2026-05-25/` and its zip) was
removed; it was a byte-for-byte second copy minus the bibliography, so it could
not be compiled anyway. Recover it from history if ever needed:

```bash
git log --diff-filter=D --oneline -- docs/arxiv_2026-05-25
git checkout <commit>^ -- docs/arxiv_2026-05-25
```
