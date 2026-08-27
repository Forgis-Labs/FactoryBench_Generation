# FactoryBench linear-probing results (rebuttal W.6)

Companion to [linear_probing_design.md](linear_probing_design.md). Records what
was run and found. All numbers reproduce from
`output/probing/qwen3_4b_rerun/results.json` + `summary_table.csv`.

**This is the audited re-run** (2026-07-25) after a pipeline review. See
"Changes from the first run" for what moved and why.

## What was run

- **Model:** `Qwen/Qwen3-4B` (open-weight panel member), frozen, no fine-tuning.
- **Inputs:** the *exact* faithful eval prompts the zero-shot panel saw
  (`output/test_eval/prompts/level{1,4}`), median ~9.9k tokens (up to 16.6k).
- **Compute:** 1× NVIDIA L4 (24 GB) on GCP (`forgisprova`), stopped after the run.
  Qwen3's grouped-query attention needs the memory-efficient SDPA kernel for the
  16k prompts; that kernel requires Ampere+ (a T4/Turing falls back to the MATH
  kernel and OOMs). L4/A100 work.
- **Read-outs:** residual stream at every layer (0=embedding … 36), two positions —
  last prompt token (`last`) and mean-pooled over the time-series token span (`mean`).
- **Probe:** logistic regression on the top-256 PCs of the (standardised)
  activations. The best layer is chosen by **episode-grouped CV on train** and its
  **held-out test** accuracy reported (so layer choice never peeks at test). MLP
  (1×128 ReLU) as a non-linear reference.
- **Controls:** (i) **selectivity** = probe acc − shuffled-label-probe acc; (ii)
  **raw-input baseline** = same probe on per-channel signal statistics (timestamp
  stripped, so no time leakage); (iii) episode-disjoint train/test; bootstrap 95% CIs.
- **Causal guard:** a startup self-test verifies attention is causal (position-k
  predictions are invariant to removing later tokens). It **PASSED** (Δ=0.0).
- **Behavioural `a`:** the same model is asked the concept directly — anomaly as
  yes/no, fault_family as a fixed A–F multiple choice over exactly the probe's six
  classes (scored against `fault_id`). The original L4 question is **stripped** first
  so the model answers only the clean question. The model is allowed to reason
  (chain-of-thought); we take its committed answer when it gives one, and for items
  where it reasons without committing we read its choice **parse-free** from the
  logits over the option tokens after a "therefore the answer is" cue — so **every
  item is scored (parse rate 1.0), none discarded.** For fault_family the model never
  produced a clean standalone letter in 160 tokens (committed 0%, all via the forced
  read) and its choices collapse onto two classes (`external disturbance` ×29,
  `unstable mounting` ×24 of 61) — a near-degenerate default, mirroring anomaly's
  always-"yes".

## Results (audited re-run)

| Concept (level) | n | chance / test | best linear (cv-layer) | 95% CI | selectivity | MLP | raw-input | behavioural `a` |
|---|---|---|---|---|---|---|---|---|
| **anomaly** (L4) | 600 | 0.577 / 0.623 | **0.834** (L2 `last`) / 0.848 (L1 `mean`) | [0.77,0.90] | 0.34 / 0.27 | 0.83 | 0.834 | **0.371** (pred-pos 1.00, parse 0.99) |
| **fault_family** (L4) | 243 | 0.309 / 0.328 | **0.492** (L1 `last`); 0.574 at L31 (argmax) | [0.38,0.62] | 0.16–0.38 | 0.53 | 0.279 | **0.16** (all items scored) |
| **task_phase** (L1) | 422 | 0.358 / 0.355 | **1.000** (L3 `last`) / 0.589 (L19 `mean`) | [1.0,1.0] | 0.87 | 1.00 | 0.523 | — |
| **source_dataset** (L4, +control) | 600 | 0.593 / 0.557 | **1.000** (L1 `last`) | [1.0,1.0] | 0.83 | 1.00 | 1.00 | — |

**Positive control passes.** `source_dataset` (which robot/dataset an episode
comes from) is trivially decodable at **1.000** (selectivity 0.83, raw-input also
1.00). This confirms the rig recovers a concept that is genuinely present, so the
near-chance results elsewhere (and the raw-vs-model gaps) are real signal, not a
broken probe.

## Findings

### 1. Read-out failure is real — and the mechanism is now explicit
Anomaly is **linearly decodable at ~0.83–0.85** (selectivity 0.27–0.34; shuffled
control near chance, so not probe memorisation), yet asked the clean yes/no
question the model scores **0.371 with `pred_pos_rate = 1.00`** — it answers
"anomaly: yes" to *every* test episode (99% parsed). The gap `p − a ≈ 0.47`.

The representation cleanly separates healthy from faulty; the model's *policy*
does not — it collapses to always-"yes". This is exactly Reviewer W.6's
distinction: the concept **is present in the representation but the model cannot
use it to answer**. (The first run reported `a=0.30` "below chance"; that number
was inflated downward by a prompt-construction confound — see below. The corrected
`a=0.371` with pred-pos 1.00 tells the same read-out-failure story, but the
mechanism — a degenerate always-yes policy — is now explicit and defensible.)

### 2. The model adds genuine representational value for fault type
`fault_family` is decodable at **0.49 (cv-selected) up to 0.57 (deep layer L31)**
vs a **raw-signal baseline of 0.28 (≈ chance)**, selectivity 0.16–0.38, MLP ≈ linear
(0.53). Because the raw numbers alone only reach chance, this reflects computation
the model performs over the signal, and it emerges in **deep layers** — a computed,
not surface, concept. The model's L4 problem is again downstream of having the info.

### 3. Depth signature separates surface from computed concepts
Anomaly is decodable immediately (best at shallow layers, ~0.71 already at the
embedding — consistent with it being partly a raw-statistics property), whereas
fault type only becomes decodable deep (L31). `fig_layerwise.pdf` shows the
trajectories; directly comparable to the depth analyses in *Words in Motion*.

## Honest caveat on task_phase (leakage — not a benchmark flaw, not evidence)
`task_phase` probes at 1.000, but from the **question wording**, not the signal:
the sole L1 template ("isolate the *lift/grasp/release…* … at which timestamp?")
names the phase in the prompt, so `phase_name` maps 1:1 onto text. It is 1.0 at a
**shallow last-token** layer and only 0.59 from the **mean-pooled time-series**
tokens — the signature of lexical availability, not time-series understanding. The
benchmark's actual answer (a *timestamp*) is not leaked. We treat task_phase as a
rig check only, and lean the claims on anomaly and fault_family.

## Changes from the first run (the audit)
1. **Behavioural prompt confound [fixed].** The yes/no question had been appended
   *after* the original L4 prompt, whose "…if yes, identify the root cause…" primes
   a "Yes,…" answer. Now the original question is stripped; `pred_pos_rate` is
   logged. Corrected `a` = 0.371 (was 0.30).
2. **Best-layer selection [fixed].** Was max **test** accuracy over ~37 layers × 2
   read-outs (optimistic). Now selected by grouped-CV on train, test reported.
   Anomaly 0.86→0.83, fault_family 0.61→0.49 under honest selection.
3. **Positive control [fixed].** Labels are now integer-encoded before probing,
   fixing the string-label crash (sklearn's MLP early-stopping called `np.isnan`
   on string predictions). `source_dataset` now probes to 1.000 and validates the
   rig. A per-concept guard also keeps one concept's failure from aborting the run.
4. **Causal attention [verified, prior concern retracted].** An over-strict bf16
   self-test had suggested attention might be bidirectional; direct testing showed
   the mask patch is a **no-op (0.0 logit diff) and attention is causal**. The
   representations — including the first run's — were computed causally and are valid.
5. **Minor:** report test-split chance; left-truncate to preserve the prompt end.

## Files
- `output/probing/qwen3_4b_rerun/results.json` — full per-layer numbers, both read-outs.
- `output/probing/qwen3_4b_rerun/summary_table.csv` — the table above.
- `fig_probe_vs_answer.pdf` — probe acc vs behavioural acc (anomaly far above diagonal).
- `fig_layerwise.pdf` — per-layer linear acc + selectivity per concept.
- `fig_linearity.pdf` — linear vs MLP (≈ equal → concepts are linearly encoded).

## Reproduce
```
python scripts/probing/run_probing.py --repo . --model Qwen/Qwen3-4B \
  --out output/probing/qwen3_4b --n-per-concept 600 --max-length 16384 --mlp --behavioural
python scripts/probing/plot_probes.py --results output/probing/qwen3_4b/results.json \
  --out output/probing/qwen3_4b
```
Needs an Ampere+ GPU (L4/A100/H100) for the ~16k-token prompts. Activations cache
under `output/probing/acts_cache/` so re-probing is instant.
