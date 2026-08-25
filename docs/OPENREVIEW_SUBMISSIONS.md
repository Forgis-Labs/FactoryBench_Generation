# OpenReview submission sheet

Everything the three workshop forms ask for. Generated 2026-08-25; abstracts are
extracted from `docs/paper/_workshop_<venue>.tex`, so they match the uploaded
PDFs exactly. Regenerate after any abstract edit.

**Submit in deadline order.** PUDM is first and is an early-morning UTC
deadline, so upload it the day before.

---

## Status: submitted

All three submitted from the `2026-08-25` build. Correspondence author: Jonas
Petersen.

**Supplementary material was deliberately not uploaded** for any of the three.
The paper already carries the code and dataset links on page 1, so the source
zip would add nothing a reviewer cannot already reach. The
`*-arxiv-source.zip` files stay in `docs/dist/` for arXiv, where a flat source
upload is required rather than optional.

Nothing below needs action unless a paper is revised; then rebuild with
`make -C docs archives` and regenerate this sheet so the abstracts stay in
step with the PDFs.

---

## Shared across all three

### Title
```
FactoryBench: Evaluating Industrial Machine Understanding
```

### Authors, in order

| # | Name | Affiliation | OpenReview ID |
|---|---|---|---|
| 1 | Yanis Merzouki *(corresponding)* | ETH Zurich; Forgis | `~TODO` |
| 2 | Coral Izquierdo | Forgis; UC3M | `~TODO` |
| 3 | Matei Ignuta-Ciuncanu | Imperial College London; UC Berkeley | `~TODO` |
| 4 | Marcos Gomez-Bracamonte | ETH Zurich; KTH | `~TODO` |
| 5 | Riccardo Maggioni | Forgis | `~TODO` |
| 6 | Alessandro Lombardi | Forgis | `~TODO` |
| 7 | Camilla Mazzoleni | Forgis | `~TODO` |
| 8 | Federico Martelli | Forgis; ETH Zurich | `~TODO` |
| 9 | Balazs Gunther | ETH Zurich | `~TODO` |
| 10 | Jonas Petersen *(equal senior)* | ETH Zurich; Forgis | `~TODO` |
| 11 | Philipp Petersen *(equal senior)* | University of Vienna | `~TODO` |

Correspondence: `jep79@cantab.ac.uk`.

> **Two things to settle before you paste this in.**
> 1. **OpenReview profile IDs.** The form wants `~Firstname_Lastname1` per
>    author. I do not have them; look each up, or add co-authors by email and
>    let OpenReview resolve them.
> 2. **Jonas will be the correspondence author.**

### Confirmation checkboxes

| Field | Answer | Why |
|---|---|---|
| Email Sharing | **Yes** | Routine; emails go to Program Chairs only. |
| Data Release | **Yes** | Non-archival venues; names released only on acceptance. |
| Dual Submission | **Yes, confirm** | See the note below before ticking. |

**On dual submission.** FactoryBench is under review at the NeurIPS 2026 main
conference. Every one of these three workshops explicitly permits work under
review elsewhere and forbids only work already *published or accepted*, so the
confirmation is accurate today. Two caveats:

- The wmphysai CFP adds that work *already published or presented at the
  NeurIPS 2026 main conference* is ineligible. Under review is not that. But if
  the main-conference decision lands before the workshop, re-read that clause.
- A deanonymised preprint of the extended paper is on arXiv
  (`arXiv:2605.07675`). Preprints are permitted and do not breach double-blind
  policy at NeurIPS venues, but reviewers may find it. Nothing to do; just know.

### Reviewer nomination

Required by PUDM, and wmphysai separately requires that one author agree to
review. Nominee needs a peer-reviewed publication in ML/robotics/CV and gets up
to 3 papers, never their own. Pick a senior author who will be available in
September, and supply full name, email, and `~Profile_ID1`.

### License

Pick the most permissive option the form offers that is still non-commercial if
such a choice exists; the repository ships under the Forgis Source Code License
(Non-Commercial) and the data under CC BY-NC 4.0. If the only options are
CC BY 4.0 or CC BY-SA 4.0, note that these govern the *submission PDF* on
OpenReview, not the released artefacts, and either is fine.

---

## Physical Understanding for Decision-Making: Bridging Foundation Models and Reliable Agents (PUDM)

- **Deadline:** 2026-08-26, 08:00 UTC
- **Portal:** https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PhysUnderstand
- **CFP:** https://sites.google.com/view/neurips-2026-workshop-pudm/submit
- **Limit:** 9 main pages; references and appendix excluded
- **This submission:** 8 main pages + references + appendix (43 total)

### Keywords
```
industrial robotics, physical understanding, decision-making, benchmark, large language models, counterfactual reasoning, linear probing, time-series reasoning, fault diagnosis, evaluation
```

### TL;DR
```
FactoryBench evaluates a model's ability to understand machines and specifically robots like an experienced operator would by using Q&A based evaluation across 4 abstraction levels.
```

### Abstract
```
Industrial robots emit dense multivariate telemetry that determines their operating state, and acting on a fault requires both reading that state and knowing the procedure that resolves it. These are separate abilities, and they are usually measured separately: perception benchmarks stop at the reading, while control benchmarks score the resulting action without isolating which of the two failed. We introduce FactoryBench, a benchmark that measures both on the same episodes from real industrial robots, so that physical perception and physically grounded decision-making can be told apart. Q&A pairs are organized along four levels, physical state estimation, intervention, counterfactual reasoning, and engineering decision-making, and span five answer formats: four are scored deterministically and free-form answers by an LLM-as-judge voting protocol. We release FactoryWave (a dense, multitask, multivariate sensor dataset from a UR3 cobot and a KUKA KR10 industrial arm, with counterfactual episodes recorded on hardware by re-executing a task under held-fixed conditions to approximate the do-operation) and build FactoryBench as over 70k Q&A items grounded in roughly 15k normalized episodes from FactoryWave, AURSAD, and voraus-AD. Evaluating six frontier models, none exceeds 50% (chance-corrected) on the perception-side levels, and the ranking reshuffles entirely at the decision level: the leader on the first three levels falls to the bottom of the fourth. Linear probes show this is not one deficit but two, recovering fault presence from a frozen model's activations at 0.83 where its own answer is below chance (a read-out failure), while fault type is no more decodable than from an untrained network (a representational one). Physical understanding and physically grounded decision-making are dissociable, and current models fail at both in different ways.
```

### Files

- **PDF:** `docs/dist/factorybench-physunderstanding-workshop-2026-08-25.pdf`
- **Supplementary (optional):** `docs/dist/factorybench-physunderstanding-workshop-2026-08-25-arxiv-source.zip`

  The source zip is fully anonymous and well under the 100 MB cap. It is
  optional; upload it only if you want reviewers to see the LaTeX. Do **not**
  upload `factorybench-physunderstanding-workshop-2026-08-25-full.zip` — that one is for your records.

---

## 8th Robot Learning Workshop: Is Physical AI Going Zero-Shot? (WRL)

- **Deadline:** 2026-08-26, 23:59 AoE (= 2026-08-27, 11:59 UTC)
- **Portal:** https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/WRL
- **CFP:** https://www.robot-learning.ml/2026/submissions/
- **Limit:** 6 pages; references excluded
- **This submission:** 6 main pages + references (10 total)

### Keywords
```
zero-shot transfer, robot foundation models, industrial robotics, cross-embodiment, benchmark, large language models, time-series reasoning, fault diagnosis, evaluation, physical AI
```

### TL;DR
```
A zero-shot benchmark on real industrial robot telemetry: six frontier models stay under 50% chance-corrected where a human expert reaches 0.96-0.98, so zero-shot transfer to an unseen machine is not yet actionable.
```

### Abstract
```
Robot foundation models are increasingly deployed zero-shot: downloaded pretrained and pointed at a machine they were never trained on. We ask what such a model understands about a physical system it is seeing for the first time, and introduce FactoryBench, a zero-shot benchmark over dense industrial robot telemetry. Q&A pairs span four levels (state estimation, intervention, counterfactual reasoning, and engineering decision-making) and five answer formats, four scored deterministically and free-form answers by an LLM-as-judge protocol. We release FactoryWave (a dense multitask sensor dataset from a UR3 cobot and a KUKA KR10 industrial arm) and build FactoryBench as over 70k items grounded in roughly 15k episodes from FactoryWave, AURSAD, and voraus-AD. Across six frontier models, none exceeds 50% (chance-corrected) on the structured levels, two fail to beat a linear-regression baseline on state estimation, and the ranking reshuffles entirely on decision-making, with the leader on the first three levels falling to the bottom of the fourth. A human expert reaches 0.96-0.98 on the same items, so the headroom is real rather than an artifact of unanswerable questions. Zero-shot transfer to an unseen industrial machine is not yet reliable enough to act on.
```

### Files

- **PDF:** `docs/dist/factorybench-robotlearning-workshop-2026-08-25.pdf`
- **Supplementary (optional):** `docs/dist/factorybench-robotlearning-workshop-2026-08-25-arxiv-source.zip`

  The source zip is fully anonymous and well under the 100 MB cap. It is
  optional; upload it only if you want reviewers to see the LaTeX. Do **not**
  upload `factorybench-robotlearning-workshop-2026-08-25-full.zip` — that one is for your records.

---

## World Models in Physical AI (WM_PAI)

- **Deadline:** 2026-08-29 AoE (= 2026-08-30, 12:00 UTC)
- **Portal:** https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/WM_PAI
- **CFP:** https://www.worldmodels-physicalai.com/cfp.html
- **Limit:** 8 pages; references excluded. Appendix not addressed, so none shipped
- **This submission:** 8 main pages + references (11 total)

### Keywords
```
world models, physical AI, industrial robotics, causal hierarchy, intervention, counterfactual reasoning, benchmark, large language models, time-series reasoning, evaluation
```

### TL;DR
```
A benchmark testing whether LLMs hold an implicit world model of a real industrial robot: six frontier models stay under 50% chance-corrected, and the ranking reverses entirely on decision-making.
```

### Abstract
```
We introduce FactoryBench, a benchmark for evaluating whether time-series models and LLMs hold an implicit world model of a physical system: an internal representation of its state and dynamics accurate enough to predict how the system evolves under an intervention and to reason about it counterfactually. Q&A pairs are organized along four levels, state estimation, action-conditioned forward prediction, counterfactual rollout, and decision-making, instantiating Pearl's ladder of causation, and span five answer formats: four are scored deterministically and free-form answers are scored by an LLM-as-judge voting protocol. We propose a scalable Q&A generation framework built around structured templates, release FactoryWave (a dense, multitask, multivariate sensor dataset from a UR3 cobot and a KUKA KR10 industrial arm), and construct FactoryBench as a benchmark of over 70k Q&A items grounded in roughly 15k normalized episodes from FactoryWave, AURSAD, and voraus-AD. Zero-shot evaluation of six frontier LLMs shows that no model exceeds 50% (chance-corrected) on state, intervention, or counterfactual reasoning, and that the ranking reshuffles entirely on decision-making, with the leader on the first three levels falling to the bottom of the fourth: current models' implicit world models are far from adequate for physically grounded, closed-loop decision-making.
```

### Files

- **PDF:** `docs/dist/factorybench-wmphysai-workshop-2026-08-25.pdf`
- **Supplementary (optional):** `docs/dist/factorybench-wmphysai-workshop-2026-08-25-arxiv-source.zip`

  The source zip is fully anonymous and well under the 100 MB cap. It is
  optional; upload it only if you want reviewers to see the LaTeX. Do **not**
  upload `factorybench-wmphysai-workshop-2026-08-25-full.zip` — that one is for your records.

---

## Before each upload

```bash
make -C docs archives          # rebuild all three, dated
```

Then, per venue, confirm on the PDF you are about to upload:

- page 1 reads `Anonymous Author(s)` and the code link is `anonymous.4open.science`
- `pdfinfo` shows empty `Author` and `Title`
- the body ends on or before the venue's page limit (references start on their own page)

`make_archive.sh` already fails the build on undefined citations, missing
figures or any LaTeX error, so a produced archive has passed those checks.
