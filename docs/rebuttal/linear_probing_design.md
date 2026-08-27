# FactoryBench linear-probing experiment (rebuttal W.6)

## Motivation

Reviewer W.6 correctly notes that FactoryBench is currently a purely
*behavioural* evaluation: the model is graded on its final answer, and we
cannot tell whether a wrong answer is caused by

- (a) the relevant machine concept being absent from the model's internal
  representation, or
- (b) the concept being *present* in the representation but the model not
  being able to translate it into a correct answer through its output head.

Distinguishing (a) from (b) is critical for interpreting the L1--L3 collapse
and the L4 signal-vs-protocol dissociation identified in Section 5. This
document proposes a targeted linear-probing study that can be executed as
an appendix experiment for the rebuttal.

## Setup

### Models

Open-weight only (activations are accessible):

- **Qwen3-4B** — small, cheap, already in the panel. Primary probing target.
- **Qwen3-235B** — larger, in the panel; probe if compute allows, otherwise
  reserved for the extended version.

Closed-weight panel members (Claude Sonnet 4.6, GPT-5.1, DeepSeek V3.2,
Mistral Large 3) are excluded because we do not have activation access.

### Probing protocol

For each item in a held-out probe set:

1. Feed the full FactoryBench prompt (question + time-series context) to the
   frozen LLM. No fine-tuning, no prompt engineering.
2. At the final token position, extract the residual-stream activation from
   every transformer layer $\ell \in \{1, \dots, L\}$ (Qwen3-4B has $L=36$,
   Qwen3-235B has $L=94$).
3. Train a per-layer, per-concept **linear** probe (logistic regression with
   L2 regularisation, chosen by 5-fold CV on the training set) to predict the
   ground-truth concept from the layer-$\ell$ activation. Also train an MLP
   probe (1 hidden layer, 512 units, tanh) as a non-linear reference.
4. Report probe accuracy on a held-out probe test split, together with the
   model's own zero-shot accuracy on the same items.

Guarding against confounds:

- Same episode never appears in probe train and probe test.
- Probe train / test are disjoint from the FactoryBench train / val / test
  splits.
- Frozen random-init baseline model to rule out artefacts from
  time-series-token positional bias.

### Concepts probed (one per FactoryBench level)

| Level | Concept the probe predicts | Ground-truth source | Format |
|-------|---------------------------|---------------------|--------|
| L1 (state) | Current task phase index (10-way, PnP) | `task_phase` label at prediction timestep | 10-way classification |
| L2 (intervention) | Sign of post-event delta on the target channel over the next 5 steps | Computed from labelled fault-injection timestep | Binary |
| L3 (counterfactual) | Sign of the counterfactual delta between paired (baseline, fault) runs | Computed from CF pair, event fixed | Binary |
| L4 (decision) | Root-cause label from the 27-way FactoryWave fault catalogue | `fault_id` in provenance metadata | 27-way classification |

Each probe uses items drawn *only* from the corresponding level's Q\&A pool,
so the prompt seen by the model at probe time matches the prompt seen at
eval time up to the final answer token.

### Sizing

Probe train / test: 800 / 200 items per level (balanced across templates).
Total forward passes needed: $4 \times 1000 = 4000$ per model. On an A100
this fits in about an hour for Qwen3-4B and roughly a day for Qwen3-235B.

## What the results tell us

Let $p_\ell^{(c)}$ denote linear-probe accuracy at layer $\ell$ for concept $c$
and $a^{(c)}$ the model's zero-shot answer accuracy on the same items (mapped
to the same concept via the ground-truth linking table above). Three regimes
are diagnostic:

- **Extraction failure** — $\max_\ell p_\ell^{(c)} \gg a^{(c)}$. The concept is
  in the representation but the output head does not surface it. Argues for
  tool-augmented or fine-tuned decoding on top of a frozen backbone.
- **Representation failure** — $\max_\ell p_\ell^{(c)} \approx a^{(c)}$, both
  near chance. The concept is not linearly decodable at any layer; pretraining
  has not exposed the model to the underlying inductive bias.
- **Non-linear encoding** — MLP probe $\gg$ linear probe. The concept is
  present but non-linearly encoded; a stronger read-out (or targeted
  fine-tuning) can recover it.

Additionally, layer-by-layer probe accuracy gives a *representation
trajectory* — where in the depth the concept first becomes decodable, which
is directly comparable to the geometric analyses in *Words in Motion*
(Grigorev et al., 2024) that the reviewer flagged.

## Deliverables for the rebuttal

1. **Per-level probe-accuracy vs answer-accuracy scatter** for Qwen3-4B: one
   figure with four points (one per level), quadrant labels showing which
   levels are extraction-limited vs representation-limited.
2. **Per-layer probe-accuracy curves** for the L2 and L3 concepts on Qwen3-4B
   (the levels where the paper's collapse is most visible).
3. **One-paragraph writeup** in the Discussion / Limitations section pointing
   at the extraction vs representation split as the natural next-generation
   FactoryBench benchmarking axis.

## Runnable skeleton

Implementation checkpoints, mapped to files in this repo:

1. `scripts/probing/dump_activations.py` — load Qwen3-4B via
   `transformers.AutoModelForCausalLM`, iterate over probe items, save
   per-layer residual-stream activations at the last non-padding token to
   Parquet.
2. `scripts/probing/build_labels.py` — for each item, materialise the
   ground-truth concept label from the linking table above.
3. `scripts/probing/train_probes.py` — per (layer, concept), fit
   `sklearn.linear_model.LogisticRegression(C=1e-2, max_iter=1000)` with
   5-fold CV; report train / CV / test accuracy.
4. `scripts/probing/plot_probes.py` — emit the two figures above under
   `output/probing/`.

Estimated end-to-end wall time on a single A100: ~4 hours for Qwen3-4B.
Notebook and CLI both work; CLI preferred for reproducibility.
