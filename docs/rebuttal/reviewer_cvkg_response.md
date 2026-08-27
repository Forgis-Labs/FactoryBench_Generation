# Response to Reviewer cVKG

We thank the reviewer for the careful reading, for calling out FactoryWave's industrial
scope and the causal-hierarchy design, and for the sharp, actionable weaknesses. All
three concerns map directly to analyses we have added since submission, and we address
them with concrete numbers below. We also acknowledge the two additional limitations
the reviewer flags and will surface them explicitly in the revised text.

## W1 / Q1: Representation bottleneck vs reasoning deficit (TS encoder and tool-augmented agent baselines)

We agree that text-serialised zero-shot is one interface among several and that scoping
the paper's claims to that interface is the right framing. The submission already
includes a specialised time-series baseline that isolates the representation question,
and we have since added a tool-augmented agent that pairs a TSFM and other specialised
tools with the LLM. Both directly quantify the gap the reviewer describes.

**Chronos-Bolt (Appendix B, in the submission).** On the three forecasting templates
of the test split, the pretrained 200 M-parameter Chronos-Bolt reaches +51.5% / +46.9%
/ +49.7% signed chance-corrected accuracy on L1.7, L2.4, L2.5, competitive with (or
above) every LLM in the panel on the forecasting slice. The remaining 18 of 21
templates (classification, ranking, counterfactual reasoning, free-form remediation)
are structurally outside any current univariate TSFM's scope. This localises the
representational gap: it exists on the templates where forecasting is the right tool,
and it does not extend to the reasoning-heavy templates.

**Tool-augmented LLM agent (new Appendix C, since submission).** We built a ReAct-style
agent driven by GPT-5.1 with four tools: `forecast` (Chronos-Bolt, i.e. the same
non-LLM TSFM baseline used above), `run_python` (sandboxed numpy/scipy for windowed
searches, derivatives, pattern matching), `signal_stats` (per-channel statistics), and
`retrieve_manual` (FAISS RAG over 9 vendor PDFs, 1,762 chunks). The agent decides which
tool (if any) to invoke, executes it, observes the output, and emits a final answer in
the item's required format. Evaluated on a stratified random subset of the test split
(200 items per level, seed 42, same-item comparison against zero-shot GPT-5.1):

| Level                          | GPT-5.1 zero-shot | GPT-5.1 agent | $\Delta$ |
| ------------------------------ | ----------------: | ------------: | -------: |
| L1 (State, signed CC)          |             15.9% |         38.8% | +22.9 pp |
| L2 (Intervention, signed CC)   |              1.3% |         15.2% | +13.8 pp |
| L3 (Counterfactual, signed CC) |             18.2% |         24.9% |  +6.6 pp |
| L4 (Decision, raw judge)       |             16.2% |         33.2% | +17.0 pp |

_Note: the GPT-5.1 numbers may differ slightly from those in the original paper because we removed the clipping in the scoring function, prioritising a signed chance correction over keeping scores within a strict [0, 1] interval._

The agent tool-usage split (247 `run_python` calls, 109 `forecast` calls, 11
`retrieve_manual`, 7 `signal_stats` on L1 alone) confirms the tools are actually
invoked rather than ignored. Two conclusions bear on the reviewer's point. First,
pairing a TSFM (and other specialised tools) with the LLM does close a substantial
share of the gap: the representational bottleneck is real, and it is quantifiable at
+6 to +23 pp per level. Second, even the tool-augmented composition remains short of
the strongest zero-shot model in the panel on L1 to L3, so the residual gap is not
purely a matter of tool access.

**Linear probing (new Appendix F).** To address the reviewer's decomposition of
representation vs. reasoning at the activation level, we ran a **linear-probing
study** on a subset of the benchmark prompts. We
probe **Qwen3-4B** (frozen, open weights, long-context) on the exact prompts the
zero-shot panel saw, training $L_2$ logistic-regression probes on the top-256 PCs
of the residual stream at every layer, with the layer chosen by episode-grouped
3-fold CV (never the test-set argmax). Four controls guard the read-out:
shuffled-label selectivity, a raw-signal baseline (no model), a random-init
control (same architecture, untrained), and episode-disjoint splits with bootstrap
intervals over three seeds.

The clearest instance of the read-out / representation decomposition is the
**fault family** probe (asking what anomaly is present, limited to six unique classes for simplification of analysis). We report four aligned numbers on the same
held-out items:

- **Probe accuracy** $p = 0.45$ at the CV-selected layer (three-seed mean
  $0.45 \pm 0.07$), above the $1/6 \approx 0.17$ chance;
- **Random-init control** $0.44$: a same-architecture network with untrained
  weights decodes fault family as well as pretrained Qwen3-4B, tracking the
  trained probe at every depth;
- **Raw-signal statistics baseline** $0.30$, above chance but well below the probe;
- **Behavioural accuracy** $a = 0.13$, around the $1/6$ chance, even with a
  chain-of-thought budget.

Fault family therefore sits at the least-favourable point of the representation /
read-out spectrum: some structure is in the signal (raw statistics reach $0.30$),
but the trained model's decodability ($0.45$) is not a product of pretraining,
since the random-init network matches it ($0.44$), and on top of that the model
cannot report the concept behaviourally ($0.13$). The low score is thus both a
representational and a read-out failure, which the probing localises cleanly.

The other probe generalises the decomposition: **fault presence** (binary) is a
pure read-out failure, decodable at $p = 0.83$ from the frozen activations while
the model's behavioural answer stays around chance, a miscalibrated
anomaly-predicting policy at inference time. We are extending the probe set to
more template families.

## W2 / Q2: Reliability of the L4 LLM-as-judge protocol

We have added a full **inter-judge reliability study** since submission, which we agree
should be part of the paper. The L4 ensemble uses three judges from different vendors
(GPT-5.1, Claude Sonnet 4.6, DeepSeek V3.2), each scoring every free-form answer on
the {0, 0.5, 1} rubric, aggregated by the median of the three votes. Pooling every L4
item for which all three judges returned a valid verdict across the six-model evaluee
panel ($n = 6{,}617$ answers, three verdicts each), we report:

- **Unanimous agreement on 87.5% of items.** The three judges assign the same score.
- **Pairwise exact agreement 90 to 93%** across the three judge pairs.
- **Fleiss $\kappa = 0.60$** across the three judges and interval Krippendorff's $\alpha = 0.78$, both in the "substantial" range; pairwise quadratic-weighted $\kappa$ is $0.74$--$0.81$.
- **Independent-grader audit on a stratified 100-item sample:** an independent grader
  (not one of the three judges) matches the panel median on 95% of items with
  quadratic-weighted $\kappa = 0.79$. The disagreements concentrate on partial-credit
  0.5 assignments rather than on the 0 vs 1 boundary, so the panel-vs-independent
  gap is on the calibration of partial credit, not on the correct-vs-wrong decision.

The judge protocol therefore does not carry the circularity the reviewer is (rightly)
alert to: the L4 collapse is stable to changing judge composition, to reducing to any
single judge, and to substituting an independent grader.

## W3 / Q3: Human baseline and question solvability

We agree that a human ceiling is the strongest complement to the automatic metric
and that it does two things at once: it certifies solvability and it calibrates
the absolute-difficulty interpretation of the low model scores. We have **begun a
small human baseline answered by engineers** and are extending it to a stratified
sample across the four levels; the current sample is 10 to 15 items, which is too
small to report a level aggregate but confirms the items are solvable in
principle. We are extending this to a larger stratified evaluation in the next few
days (target: at least 25 items per level, answered by multiple robotics-expert
annotators already assigned, with adjudication) and will report a per-level human
ceiling in an updated appendix.

On the solvability check we can offer today, we point the reviewer to Appendix E in
the submission (**Validation of solvability**): every released template was
programmatically stress-tested against the generator's ground-truth extractor on a
1,000-episode audit before release, and Appendix E.1 (noise-substitution) shows that
the strong models genuinely read the signal at L2, L3 and L4 (scores collapse when the
time-series values are replaced with cumulative-Gaussian noise while questions and
options are held fixed). The linear-probing appendix (App. F, since submission) is
a further, representation-level solvability check: fault presence is linearly
decodable at 0.83 from the frozen model's activations even when the model's own
answer collapses to chance, so the low behavioural scores reflect a read-out gap
rather than unsolvable items.

---

We hope these responses resolve the reviewer's concerns and we are glad to run any
further analysis during the discussion period.
