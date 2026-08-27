# Yanis's original workshop drafts (reference only, not tracked)

The three PDFs he sent, preserved because they are the ONLY surviving record of
his per-venue framing: the `_workshop_<venue>.tex` sources were never pushed to
any branch of either repository, and he has since left.

| PDF | venue | shape |
|---|---|---|
| `main_wmphysai.pdf` | World Models in Physical AI | 11pp, full body, no appendix |
| `main_physunderstanding.pdf` | physical understanding | 40pp, full body + shared appendix |
| `main_robotlearning.pdf` | robot learning | 10pp, compact body |

The sources under `docs/paper/_workshop_<venue>.tex` and
`docs/workshop_tex/main_<venue>.tex` are reconstructions of these, and differ
from them deliberately in two ways:

1. **Anonymity.** All three of these PDFs print the live dataset URL
   `huggingface.co/datasets/FactoryBench/FactoryBench` on page 1, as visible
   text and as a clickable annotation, directly beneath a correctly anonymised
   code link. The reconstructions withhold it.
2. **The agent baseline.** His local `_workshop_core.tex` had dropped the
   agent-baseline sentence from the workshop conclusion; the committed core
   keeps it, now stating the measured null result (-2.9 pp, CI [-7.4, +1.6])
   rather than the retracted lift.

Do not submit these PDFs. Build from source instead:

    make -C docs archives
