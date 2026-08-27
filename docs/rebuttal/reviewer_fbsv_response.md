# Response to Reviewer Fbsv

We thank the reviewer for the sharp, technical read and for flagging six concrete
issues; every one is fair. Five are artefacts we can fix inside the discussion period,
and the sixth (linear probing) motivated an analysis we have now run. We address each
in turn.

## W.1 (and Q1): Chance-correction transform

The reviewer is right on the math. At submission we clipped each score to $[0, 1]$ for
interpretability against the regression baseline and a $100\%$ ceiling, accepting the
loss of the zero-under-random property. Having now run the aggregation both ways, we
agree that was the wrong call: the signed transform $\tilde{s} = (s - E)/(1 - E)$
separates the panel from the simple-random baseline far more visibly and restores
zero-under-random by construction. We have switched to it, recomputed every number, and
rewritten the correction section to state the per-format chance level $E$ (single-select
$1/k$; multi-select $1/2$; ranking $1/n$; tensor $1/4$).

The clearest payoff is on the main panel. With the baseline near its empirical zero, one
can read directly which models separate from the simple (non-LLM) baseline at each
level, starkest on Level 1 (signed chance-corrected accuracy):

| Model                        |    L1 |    L2 |    L3 |
| ---------------------------- | ----: | ----: | ----: |
| Claude Sonnet 4.6            | 46.8% | 25.8% | 35.8% |
| Qwen3-235B                   | 29.7% | 10.9% | 30.9% |
| Mistral Large 3              | 24.8% |  5.6% | 23.4% |
| GPT-5.1                      | 19.0% |  4.0% | 18.2% |
| Simple baseline (regression) | 12.2% | -3.0% |  1.7% |
| DeepSeek V3.2                | 10.8% |  4.1% | 12.8% |
| Qwen3-4B                     |  5.9% |  0.3% | 12.4% |

Two of the six models (Qwen3-4B and DeepSeek V3.2) fall below the simple baseline on L1
and GPT-5.1 clears it only modestly, whereas Claude Sonnet 4.6, Qwen3-235B, and Mistral
Large 3 establish a clear margin; the clipped rule compressed scores toward the top of
$[0, 1]$ and hid this. We also recomputed the noise-substitution appendix the same way;
the L2/L3 collapse under noise is if anything sharper, so we omit that table here.

## W.2: Discrepancy between Fig. 3 and Fig. 4 counts

Same cause as W.4 and W.5: late additions to FactoryWave and the generated Q\&A set
landed in the shipped test split (Figure 4) between the scoring run behind Figure 3 and
the release cut, and the panel was never rerun on the aligned counts. We have now
rescored the panel on the shipped test split, so Figure 3 and Figure 4 both refer to the
same `FactoryBench/FactoryBench:v1.0.0` snapshot with the same per-level counts, and the
revised caption states the counts and names the release tag.

## W.3: T+N horizons at 10 Hz

Thank you for spotting this. The L1 forecasting template L1.7 had a bug in the acceptance
bounds: the horizon field was written in units of the acquisition-time raw sampling rate
(about 1 kHz) rather than the released 10 Hz base signal, so the bound could refer to a
sub-100 ms horizon that no released row represents, and prompt strings like "T+6 ms" fell
out of the same mismatch. We have fixed both the horizon generation and the scorer, so
the horizon is now stated in units of the released 10 Hz base signal (row offsets of $1$
to $10$, i.e.\ $100$ to $1{,}000$ ms of wall-clock), and the prompt renders it as
$N \times 100$ ms so the visible "T+N ms" phrasing matches the row it points to. We also audited every other
template's generation and scoring path end-to-end for unit and indexing agreement. No
item was dropped and the ground-truth values are unchanged; Appendix E now defines the
horizon convention explicitly. The dataset itself should be fixed in the following days.

## W.4 and W.5: license, episode counts, Q\&A counts

Both inconsistencies came from last-minute pre-deadline cleanup (back-filling episodes,
regenerating a few items after a labelling audit, updating the HF licence header) landing
on some surfaces (paper, Croissant record, HF metadata, README) but not others. **We have
now aligned all four.** The intended licence is two-track: **MIT for the generator source
code, evaluation scripts, and tooling**, and **CC BY 4.0 for the data and benchmark
artefacts** (FactoryWave episodes, question templates, paraphrase banks, LLM-as-judge
prompts, and the full Q\&A dataset).

For W.5, the paper counts FactoryWave episodes using the counterfactual-group convention,
where each baseline plus all its signature-kernel-MMD-selected variants counts as one
group; this is the Table 1 convention (2{,}183 Normal + 6{,}509 Fault + 291 CF groups =
8{,}983). The live public-page total of 9{,}728 used a raw-episode convention that counted
CF variants separately, which does not match the paper. We are re-counting the live page
under the CF-group convention so both surfaces use the same rule. Episodes added to the
live branch after submission are independent recordings, not CF variants, so they remain
in the count; the paper still pins every number to `FactoryBench/FactoryBench:v1.0.0`. We
will make the "pinned to v1.0.0" phrasing more prominent and name the current main-branch
totals in the release section.

**Dataset viewer fix.** The HF viewer failure has been fixed.

## W.6: Behavioural evaluation only (linear probing)

We agree this is an important complement, and in direct response to the reviewer we have
run a **linear-probing study** (Appendix F) on a subset of the benchmark prompts covering
an L2 anomaly-recognition template restricted to the six most frequent injected fault
families (so class support per fold is comparable) and the L1 anomaly-presence template
(binary). We probe **Qwen3-4B** (frozen, open weights, long-context) on the exact prompts
the zero-shot panel saw, training $L_2$ logistic-regression probes on the top-256 PCs of
the residual stream at every layer, with the layer chosen by episode-grouped 3-fold CV
(never the test-set argmax). Four controls guard the read-out: shuffled-label
selectivity, a raw-signal baseline (no model), a random-init control (same architecture,
untrained), and episode-disjoint splits with bootstrap intervals over three seeds.

The clearest instance of the read-out / representation decomposition the reviewer asks
for is the **fault family** probe (six-way). We report four aligned numbers on the same
held-out items:

- **Probe accuracy** $p = 0.45$ at the CV-selected layer (three-seed mean
  $0.45 \pm 0.07$), above the $1/6 \approx 0.17$ chance;
- **Random-init control** $0.44$: a same-architecture network with untrained weights
  decodes fault family as well as pretrained Qwen3-4B, tracking the trained probe at
  every depth;
- **Raw-signal statistics baseline** $0.30$, above chance but well below the probe;
- **Behavioural accuracy** $a = 0.13$, around the $1/6$ chance, even with a
  chain-of-thought budget.

Fault family therefore sits at the least-favourable point of the representation /
read-out spectrum: some structure is in the signal (raw statistics reach $0.30$), but the
trained model's decodability ($0.45$) is not a product of pretraining, since the
random-init network matches it ($0.44$), and on top of that the model cannot report the
concept behaviourally ($0.13$). The low score is thus both a representational and a
read-out failure, which the probing localises cleanly.

The other probe generalises the decomposition: **fault presence** (binary) is a pure
read-out failure, decodable at $p = 0.83$ from the frozen activations while the model's
behavioural answer stays around chance, a miscalibrated anomaly-predicting policy at
inference time. We are extending the probe set to more template families, and the
Words-in-Motion pointer is well taken: we cite it in the probing appendix as related work
on linearly encoded concepts in trajectory representations.

## Additional work since submission

As an extra line of work since submission, we built a deliberately simple tool-augmented
agent: a ReAct-style GPT-5.1 that can call a time-series foundation model (Chronos-Bolt),
sandboxed numpy/scipy, per-channel signal statistics, and a vendor-manual retriever, then
answers in the item's required format. On a stratified 200-item-per-level subset, with a
same-item comparison against zero-shot GPT-5.1:

| Level                          | GPT-5.1 zero-shot | GPT-5.1 agent | $\Delta$ |
| ------------------------------ | ----------------: | ------------: | -------: |
| L1 (State, signed CC)          |             15.9% |         38.8% | +22.9 pp |
| L2 (Intervention, signed CC)   |              1.3% |         15.2% | +13.8 pp |
| L3 (Counterfactual, signed CC) |             18.2% |         24.9% |  +6.6 pp |
| L4 (Decision, raw judge)       |             16.2% |         33.2% | +17.0 pp |

This shows that even a modest amount of tool scaffolding recovers a large share of the
gap. We read this as encouraging: FactoryBench is a productive target for agentic and
tool-use research, with clear headroom for the community to build on.

Two further additions since submission:
an **inter-judge reliability study** of the three-judge L4 panel (all three perfectly agree on
$87.5\%$ of items; Fleiss $\kappa = 0.60$, Krippendorff $\alpha = 0.78$), together with an **independent human grader** matching the panel median
on $95\%$ of a $100$-item sample ($\kappa = 0.79$), confirming the L4 scores are robust to
judge composition and to a human check; and a **human baseline answered by
engineers**, currently small but being extended to a stratified per-level sample that we
will report in the paper.

---

We hope these responses resolve the concerns, and we are glad to run any further check
during the discussion period.
