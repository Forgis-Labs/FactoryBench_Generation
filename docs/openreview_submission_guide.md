# FactoryBench: OpenReview Submission Guide

> **Scope: the NeurIPS 2026 main conference (Evaluations and Datasets track),
> submitted May 2026 and under review.** Kept as the record of that
> submission. The three non-archival workshop submissions are a separate
> matter, documented in `OPENREVIEW_SUBMISSIONS.md`, and the build system is
> documented in `BUILDING.md`.

NeurIPS 2026 Evaluations and Datasets Track.

---

## 1. Track Details

| Field | Value |
|-------|-------|
| **Portal** | <https://openreview.net/group?id=NeurIPS.cc/2026/Conference> |
| **Abstract deadline** | May 5 2026, 11:59 AM UTC |
| **Full paper deadline** | May 7 2026, 11:59 AM UTC |
| **Format** | 9 pages main + unlimited refs/appendix, single PDF (<50 MB) |
| **Style** | `\usepackage[eandd]{neurips_2026}` (Evaluations and Datasets track, anonymous mode) |
| **Review** | Double-blind |

---

## 2. OpenReview Form Fields

### Filled

| Field | Value / Status |
|-------|----------------|
| **Title** | `FactoryBench: Evaluating Industrial Machine Understanding` |
| **Authors** | All 11 profiles added (Merzouki, Izquierdo Muniz, Ignuta-Ciuncanu, Gómez-Bracamonte, Günther, Lombardi, Mazzoleni, Martelli, Maggioni, Petersen J., Petersen P.) |
| **Keywords** | `benchmark, dataset, time series, question answering, industrial robotics, causal reasoning, machine understanding, anomaly detection, LLM evaluation, robotic telemetry` |
| **TL;DR** | `FactoryBench: 70k+ Q&A items over 15k industrial robot episodes across 4 causal levels (Pearl's ladder + decision), 5 answer formats with deterministic + LLM-judge scoring, grounded in FactoryWave (UR3 + KUKA KR10).` |
| **Abstract** | See below |


**Abstract** (copy-paste for OpenReview):
```
We introduce FactoryBench, a benchmark for evaluating time-series models and LLMs on machine understanding over industrial robotic telemetry. Q&A pairs are organized along four causal levels (state, intervention, counterfactual, decision) instantiating Pearl's ladder of causation, and span five answer formats: four structured formats are scored deterministically and free-form answers are scored by an LLM-as-judge voting protocol. We propose a scalable Q&A generation framework built around structured question templates, present FactoryWave (a dense, multitask, multivariate sensor dataset collected from a UR3 cobot and a KUKA KR10 industrial arm), and construct FactoryBench as a large-scale benchmark of over 70k Q&A items grounded in roughly 15k normalized episodes from FactoryWave, AURSAD, and voraus-AD. Zero-shot evaluation of six frontier LLMs shows that no model exceeds 50% on structured levels or 18% on decision-making, revealing a wide gap between current models and operational machine understanding.
```

| Field | Value / Status |
|-------|----------------|
| **Review Mode** | `Double-blind (default; anonymized submission)` |
| **Dataset Submission** | Checked |
| **Croissant File** | `factorybench_croissant.json` uploaded |
| **Dataset URL** | `https://huggingface.co/datasets/FactoryBench/FactoryBench` (anonymous throwaway account) |
| **Dataset Large URL** | Left blank |
| **Code URL** | `https://anonymous.4open.science/r/FactoryBench` |
| **Code Submission Justification** | Left blank (code is provided) |
| **Supplementary Material** | Left blank (everything is in PDF + code URL) |
| **Primary Area** | `Datasets and Benchmarks` |
| **Contribution Type** | `Datasets and Benchmarks` |
| **Reviewer Nomination** | `~Philipp_Christian_Petersen1` |
| **License** | non-commercial two-track licence |
| **Author Acknowledgements** | All 5 boxes checked |
| **LLM Usage** | Select all that apply (confidential, not shared with reviewers) |
| **LLM Experiment** | Opt in or leave unchecked |
| **Financial Support** | Fill in if a student author needs travel support (OpenReview ID, e.g. `~Yanis_Reyan_Merzouki1`) |

---

## 3. Anonymization

The paper uses `\usepackage[eandd]{neurips_2026}`, which automatically:
- Renders "Anonymous Author(s)" in place of the `\author{}` block
- Adds the "Submitted to NeurIPS 2026. Do not distribute." footer

**What is already handled (do NOT manually remove):**
- Author names, affiliations, email, suppressed by the `eandd` option
- Code link uses `anonymous.4open.science`
- HuggingFace link uses a throwaway `FactoryBench` account (no author info)
- No acknowledgments section present
- No `\todo{}` markers in rendered PDF
- Paper checklist complete (no `\answerTODO` items)

**Note:** Only the compiled PDF is uploaded to OpenReview (not the `.tex` source), so source-only content (color names, commented-out blocks, `\author{}` raw text) is not visible to reviewers.

---

## 4. Pre-Submission Checklist

- [x] Abstract registered (May 5 deadline)
- [x] Primary Area selected and locked
- [x] Contribution Type selected and locked
- [x] Reviewer Nomination set and locked
- [x] Paper checklist complete (all items answered, no `\answerTODO`)
- [x] All `\cite{}` keys resolve (no "?" in PDF)
- [x] Bib key/year mismatches fixed (`peyrard2020ladder`, `brockmann2023vorausad`)
- [x] American English standardized throughout
- [x] Typos fixed (`dimensions`, `dependent`, `ground truth`)
- [x] Figure margins checked (all figures within `\linewidth`)
- [x] Stale TODO comments and commented-out sections removed from source
- [x] Anonymized public repo reference in Appendix L
- [ ] **Compile final PDF**, recompile after latest fixes
- [ ] **Upload PDF** to OpenReview
- [ ] Provide Dataset URL
- [ ] Provide Code URL
- [ ] Select License (non-commercial two-track licence)
- [ ] Check all 5 Author Acknowledgement boxes
- [ ] Double-check rendered PDF: no author names, no institution names, footer reads "Submitted to..."

---

## 5. Compilation

```bash
cd docs/neurips_tex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Verify after compilation:
- No overfull hbox warnings in main text area
- All figures render correctly
- All references resolve (no "?" markers)
- Footer: "Submitted to 40th Conference on Neural Information Processing Systems (NeurIPS 2026). Do not distribute."
- Page 1 shows "Anonymous Author(s)" (not real names)
- Code and dataset links visible below title
