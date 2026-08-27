# Response to Reviewer NoeB

We thank the reviewer for the careful reading and the positive assessment, and in
particular for recognizing the value of the FactoryWave release and the diagnostic
value of the four-level taxonomy. We address the two weaknesses and two questions
below. Two of the analyses we cite (the linear-probing study and the inter-judge
reliability study) were added after submission and target these concerns directly.

## W1: Template reliance and Level-1 lexical answerability

We take this seriously. We agree that Level 1 (state reading) is the most lexically
anchored level and treat it as a state-reading anchor rather than a signal-comprehension
test; the signal-comprehension claim rests on L2 to L4. The important question is
whether those higher levels genuinely require the signal for capable models, and we
tested it directly.

The paper's noise-substitution check replaces only the numerical time-series values
while keeping questions, options, and gold answers fixed, so any surviving score cannot
come from reading the signal. That check used an early, weaker panel (Claude Haiku 4.5,
DeepSeek V3.1) and doesn't necessarily imply that questions are unanswerable using the time series data, but rather that those specific models cannot properly use and analyze that data. To show that, we reran the identical check with the main-study model that performed the best on L1, Claude Sonnet 4.6, on the same original and noised prompts, and this time noticed an actual decrease in scoring.

Since submission we added a **linear-probing study** that attacks the same question
at the representation level rather than the behavioural one. We probe a frozen
open-weight panel member (Qwen3-4B) on the exact evaluation prompts, reading its
residual-stream activations at every layer, with four controls: a shuffled-label
selectivity control, a raw-signal-statistics baseline (no model), a random-init
control (same architecture, untrained weights), and episode-disjoint splits with
episode-level bootstrap intervals over three seeds. Two conclusions bear directly
on this weakness:

1. **Fault presence is genuinely in the signal, and the failure is at read-out.**
   Fault presence is linearly decodable at 0.83 from activations (and at 0.83 from
   raw signal statistics alone), yet the model's own answer to the direct question
   stays near the 0.5 chance. The concept is present in the telemetry and in the representation; what
   fails is the model's calibrated read-out, not the availability of the information.

2. **Fault type exposes a representational gap, not an unanswerable question.** Fault
   type is weakly decodable (0.45), above the raw-signal baseline (0.30) but not above
   the random-init control (0.44), and the model reports it at 0.13. The information
   is present in the signal (a random network already recovers 0.44, above the 1/6
   chance), but the trained model encodes no more of it than an untrained one, and
   cannot read it out.

Together these show the low benchmark scores are not an artifact of unanswerable
questions: the information is in the signal, and the failures range from read-out and
calibration (fault presence) to genuine representational gaps (fault type). We will add
the strong-model noise-substitution rerun above to the appendix so the signal-dependence
of L2 and L3 is documented for the main-study panel.

## W2: Baselines, including a human/expert baseline

Beyond the six-LLM panel the submission already reports a **Chronos-Bolt time-series
foundation-model baseline** (Appendix B) and since submission we have added a full
**tool-augmented LLM agent** appendix that speaks to the same "is the panel the right
comparison" question.

**Tool-augmented LLM agent (new Appendix C).** We built a ReAct-style agent driven
by GPT-5.1 with four tools: `forecast` (Chronos-Bolt, i.e. the same non-LLM TSFM
baseline used above), `run_python` (sandboxed numpy/scipy for windowed searches,
derivatives, pattern matching), `signal_stats` (per-channel statistics), and
`retrieve_manual` (FAISS RAG over 9 vendor PDFs, 1{,}762 chunks). The agent
decides which tool (if any) to invoke, executes it, observes the output, and
emits a final answer in the item's required format. Evaluated on a stratified
random subset of the test split (200 items per level, seed 42, same-item
comparison against zero-shot GPT-5.1):

| Level                          | GPT-5.1 zero-shot | GPT-5.1 agent | $\Delta$ |
| ------------------------------ | ----------------: | ------------: | -------: |
| L1 (State, signed CC)          |             15.9% |         38.8% | +22.9 pp |
| L2 (Intervention, signed CC)   |              1.3% |         15.2% | +13.8 pp |
| L3 (Counterfactual, signed CC) |             18.2% |         24.9% |  +6.6 pp |
| L4 (Decision, raw judge)       |             16.2% |         33.2% | +17.0 pp |

The agent tool-usage split (247 `run_python` calls, 109 `forecast` calls, 11
`retrieve_manual`, 7 `signal_stats` on L1 alone) confirms the tools are actually
invoked rather than ignored. Two conclusions bear on the reviewer's baseline
concern. First, pairing a TSFM (and other specialised tools) with the LLM does
close a substantial share of the gap: the representational bottleneck is real
and quantifiable at +6 to +23 pp per level. Second, even the tool-augmented
composition remains short of the strongest zero-shot model in the panel on L1 to
L3, so the residual gap is not purely a matter of tool access.

Two additions since submission speak to reliability and to the requested human
baseline:

- An **inter-judge reliability study** of the three-judge L4 panel confirms the
  scores are not an artifact of any single judge: the three judges agree unanimously
  on 87.5% of items, pairwise exact agreement is 90 to 93%, and an independent grader
  matches the panel median on 95% of a representative 100-item sample
  (quadratic-weighted kappa = 0.79).
- We have **begun a human baseline answered by engineers**. It is small for now
  (10 to 15 items), but we are extending it to a stratified sample across levels and
  will include it in the paper; it also provides an achievable-performance reference
  for the low absolute scores, particularly on L4.

## Q1: Robustness to removing expert-selected feature subsets

Removing the curation and exposing the full raw channel set generally degrades
performance, but we chose to default to those expert-selected feature subsets for two main reasons worth separating. First, and most
mechanically, the context size blows up: some source datasets expose more than 100
channels per timestep, so concatenating all of them produces contexts that are both
very long (which depresses every model and exceeds the limits of the smaller ones, making an in depth analysis impractical and way more expensive) and
dominated by channels irrelevant to the question. Second, and more important for
validity, the curated subset removes superficial shortcuts. With every channel present,
a question such as "which robot is this?" can be answered from a metadata artifact like
the number of channels available (industrial arms typically expose fewer than the
research cobot) rather than from the signal itself; a fixed per-template channel budget
instead forces the model to read and reason over the feature values. The selection is
therefore a deliberate design choice that keeps the benchmark measuring signal-grounded
reasoning, and the full per-template lists (30 channels for L1 to L3, about 20 for L4)
are released so the choice is transparent and reproducible. We are glad to quantify this
with an expert-versus-random-subset ablation on a stratified subset if the reviewer
would find it useful.

## Q2: Are the Level-4 gold protocols field-validated?

The L4 gold answers are grounded in the manufacturers' official runtime-error
documentation: we parse each robot's vendor error catalogue into error code, name,
description, and recommended recovery steps, and PhD-level robotics experts map each
injected anomaly to the most plausible controller error, authoring the protocol
directly in the same structure where a physical fault triggers no controller error.
The source is therefore field-authoritative documentation, and scoring consistency is
validated by the independent-grader audit above. They are not independently validated
by deployment trials, which we already flag in the Limitations: L4 is closer to
closed-book retrieval of vendor protocols than open-ended engineering, and the low L4
scores conflate "does the model know this manual?" with "can it reason from signal to
corrective action?" (the probing study is our first step at separating these). We will
make this provenance explicit in the L4 caption.

---

We hope these clarifications resolve the reviewer's concerns, and we are glad to run
any further analysis during the discussion period.
