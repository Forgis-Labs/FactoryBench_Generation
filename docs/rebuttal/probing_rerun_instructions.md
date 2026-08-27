# Linear-probing appendix — rerun instructions

The current probing pipeline (`scripts/probing/run_probing.py`) has methodological gaps that a NeurIPS reviewer would credibly attack. The paper prose in `docs/neurips_tex/_paper_body.tex` §`app:probing` (lines 1538–1572) makes strong claims that need to be defensible; the fixes below make the code match those claims. Apply the changes in order, rerun, and confirm the numbers match the paper before submitting.

**Environment**: `factorybench` conda env, single GPU, activations cached at `output/probing/qwen3_4b_rerun/acts_cache/`.

---

## 1. PCA + StandardScaler must be inside CV folds (leakage)

**File**: `scripts/probing/run_probing.py`, function `_probe_one_layer`
**Current bug**: `_scale_reduce(X[tr], X[te], seed)` is called **once, before** `_lr_cv`, then CV runs GroupKFold on the already-PCA'd `Xtr`. Every CV validation fold's features were shaped by a PCA that included them → CV score is upward-biased → both the chosen `C` and the CV-selected layer are picked on leaky scores.

**Fix**: Move `_scale_reduce` inside the GroupKFold loop of `_lr_cv`. For each fold: fit `StandardScaler` + `PCA(n_components=k)` on the fold's train indices only, transform both train and val with that fitted pipeline, then fit LR. After CV picks the best `C`, refit the scaler+PCA on the full train, transform test, and score.

**Consequence**: CV scores drop 1–3 pp for both concepts. The CV-selected layer for `fault_family` may move away from L1. Update the paper numbers to whatever the honest CV run produces.

---

## 2. Behavioural read-out for `anomaly` must use the current code path

**File**: `scripts/probing/run_probing.py`, function `behavioural_accuracy`
**Current state**: `results.json` shows `anomaly.behavioural = {acc: 0.371, parse_rate: 0.9934, n: 151, pred_pos_rate: 1.0}` — this schema has no `method / committed_rate / forced_rate / pred_dist / gen_tokens` field. It is the return dict of an **older parse-and-drop** implementation. The current code (which is what produced `fault_family`) returns those fields.

**Fix**: Rerun `--concept anomaly --behavioural-only` with the current `behavioural_accuracy` (CoT + forced-fallback). Verify the output JSON has all six keys (`method, committed_rate, forced_rate, gen_tokens, pred_dist`, plus `acc, n`). Do not touch the probe activations — only the behavioural readout needs to rerun.

**Consequence**: `anomaly.a` may shift by 1–2 items (the current 151/151 was 150/151 → 1 dropped by the old parser). Paper narrative of "always-yes" should stay valid unless the fresh number shows otherwise.

---

## 3. Layer 31 = 0.57 for fault_family is a test-set peek

**File**: `scripts/probing/plot_probes.py::_best`; also `scripts/probing/run_probing.py` where `argmax_test_layer` is stored.
**Current bug**: The paper promises the reported layer is the CV-max ("never the test accuracy"). But the "0.57 at layer 31" sentence in `_paper_body.tex:1559` is the argmax over 74 (layer × read-out) test scores. `_best` in `plot_probes.py` also picks by test accuracy, so `summary_table.csv` reports `best_layer=31, linear_acc=0.574` and contradicts the paper's headline of 0.492 at L1.

**Fix (code)**: In `plot_probes.py::_best`, sort by CV score, not test accuracy. Regenerate `summary_table.csv` so `best_layer` reflects CV-selected. Remove `argmax_test_*` fields from `results.json` altogether or clearly rename them e.g. `test_ceiling_layer, test_ceiling_acc` and never quote them.

**Paper action**: Once code is fixed, replace `_paper_body.tex:1559` sentence to drop the "0.57 at layer 31" clause. New wording:

> Fault \emph{type} is decodable at $p{=}\text{CV-VAL}$ (CV-selected layer $L{=}\text{L-VAL}$), against a raw-signal baseline of only $0.28$ (near its $0.31$ chance).

Where `CV-VAL` and `L-VAL` are the numbers from the fixed run.

---

## 4. Fault-type CoT budget too small — forced-fallback on 100% of items

**File**: `scripts/probing/run_probing.py`, `behavioural_accuracy(..., gen_tokens=160)`
**Current state**: `results.json → fault_family.behavioural.committed_rate = 0.0, forced_rate = 1.0, gen_tokens = 160`. The model NEVER committed a letter inside the budget — every item was scored via the mid-reasoning forced-choice fallback. `a=0.164` therefore isn't the model's self-terminated answer.

**Fix**: Set `gen_tokens=1024` (or higher). Rerun `--concept fault_family --behavioural-only`. Log `committed_rate` and `forced_rate` in the output.

**Consequence**: If `committed_rate` climbs meaningfully (say > 0.5), `a` may improve or worsen — either way, report the true self-terminated rate. If it stays near 0, note in the appendix that the model doesn't commit under a 6-way CoT ask.

---

## 5. MCQ option order must randomise per item

**File**: `scripts/probing/run_probing.py`, `FAULT_CHOICES` (module constant)
**Current bug**: Same fault always maps to the same letter A–F. Model's answer distribution `{B: 4, C: 24, D: 4, E: 29}` never picks A or F — middle-of-list positional bias not controlled.

**Fix**: For each item, shuffle the six class labels to a fresh A–F permutation with a per-item seed (e.g. `hash(item.custom_id) % 2**32`). Store the permutation in the item's reply record so scoring can invert it. This applies to both the `parse-committed` path and the `forced-choice-from-logits` path.

**Consequence**: Model's `a` may change (probably upward — the ~0.16 is partly the always-picks-C-or-E bias). The paper's "collapses onto two classes" claim depends on this — recheck the pred distribution after the fix.

---

## 6. Bootstrap CIs must resample at the episode level, not the item level

**File**: `scripts/probing/run_probing.py`, `_bootstrap_ci`
**Current bug**: Items with the same `episode_id` share a lot of signal (avg 3.31 items/episode for L1, 1.06 for L4). Sampling items with replacement pretends they're independent → intervals are ~sqrt(k)× too narrow where k = mean items per episode.

**Fix**: Group items by `episode_id`, resample episodes with replacement, use all items belonging to each sampled episode. Do this **for every reported CI**: probe `linear_acc`, `mlp_acc`, `selectivity`, raw-baseline, behavioural `a`.

**Consequence**: CIs widen — especially for anomaly on L1 where items are more clustered per episode. May tip fault_family into overlapping chance. If so, update the paper's phrasing on fault-type to "decodable above signal-only chance" without asserting stronger effect size.

---

## 7. Yes-bias for anomaly needs a control

**File**: new — add to `scripts/probing/run_probing.py`
**Current gap**: The model answers "yes" to 151/151 anomaly items. Paper reads this as "read-out failure", but no control rules out a generic yes-bias in Qwen3-4B on `"Is X present? yes/no"` prompts.

**Fix (add one control, keep it small)**: Repeat behavioural elicitation with **negated wording** on the same items: `"Is this episode healthy (no anomaly)? Answer yes or no."`. Report the resulting `a_neg` and `pred_neg_pos_rate`. If the model also answers "yes" 100% (which would mean it always says the healthy option), the yes-bias is confirmed and the paper's "read-out failure" claim survives with the yes-bias caveat noted. If it answers "no" 100% (mirroring), that's the same bias in disguise. If it varies, we're finding real signal.

**Consequence**: Add one paragraph to the appendix reporting `a_neg` and interpretation. Keep whichever wording gives the honest story.

---

## 8. Multiple seeds for headline numbers

**File**: `scripts/probing/run_probing.py`, `--seed` argparse arg
**Current state**: All headline numbers from `--seed 0`. Bootstrap CIs cover item-resample noise, not seed noise (split, PCA solver, C-selection).

**Fix**: Rerun with `--seed 1` and `--seed 2` and aggregate. Report `mean ± std` in the paper text for the four headline numbers: `anomaly.p, anomaly.a, fault_family.p, fault_family.a`.

**Consequence**: The `mean ± std` replaces the current single-point numbers. Best-case: numbers are stable, paper narrative unchanged. Realistic: `fault_family.p` may drift ± 0.03; add the std explicitly to the paper.

---

## 9. Truncation-aware mean-pool span

**File**: `scripts/probing/dump_activations.py`, function `ts_token_span`
**Current bug**: `ts_token_span(offsets[:seq_len], it.prompt)` uses char offsets from the **full** `it.prompt` but the offsets come from the **left-truncated** token sequence. For prompts > `max_length` (default 8192; some items reach 16.6k) the `\nt=` marker is truncated out and the span collapses to "mean over almost the entire retained prompt".

**Fix**: (a) Bump the default `max_length` to `16384` (or higher — Qwen3-4B natively supports 32k). (b) Add a run-time check: if `tokenizer(it.prompt, add_special_tokens=False, max_length=max_length, truncation=True)` differs from the un-truncated tokenisation, log a warning and skip the item's mean-pool read-out (still use last-token). (c) Record `max_length` in the activation cache filename (e.g. `acts_cache/level4_real_n61_ml16384.npz`) so cached activations from a different truncation regime aren't silently reused.

**Consequence**: Mean-pool numbers may shift for the longest items. Last-token numbers should be unaffected. If mean-pool improves materially, the paper's "mean-pool 0.85 for anomaly" number will be updated.

---

## 10. Add the random-init control that was promised but never run

**File**: `scripts/probing/run_probing.py` has `--random-init` / `--randomize-weights`; never invoked in the paper's headline run.
**Current gap**: `summary_table.csv` shows `randinit_acc: nan` for every concept. The paper appendix advertises three controls but only two were run.

**Fix**: Rerun `--concept anomaly --randomize-weights` and `--concept fault_family --randomize-weights` — same probe pipeline on a Qwen3-4B with **randomised weights** (same architecture, no pretraining signal). Report `randinit_acc` per concept.

**Consequence**: Add one sentence to the paper's Controls list. Expect `randinit_acc ≈ raw_signal_baseline` — that would confirm the probe isn't just extracting shape/statistics of untrained network activations.

---

## Fixes I'm NOT recommending we rerun for (paper wording alone)

These are either minor or won't move the numbers meaningfully. Not worth GPU time:

- **`n_train=182 → 180 PCs` for fault_family instead of 256**: the k=256 becomes k=180 by clamp. Rerunning doesn't change this; just note "at most 256" in the paper if we care. We chose to keep "256" for readability — trivially true for the anomaly concept.
- **`class_weight="balanced"` at training, unweighted acc at eval**: correct as-is for our purposes (comparing to model's own untied answer distribution). Report macro-F1 as a supplement in the appendix table if space allows.
- **Chance reference inconsistency (full-set vs test-split)**: unify on **test-split chance** everywhere in the paper (both anomaly's 0.62 and fault_family's 0.33). This is a paper-only edit.
- **Selectivity=0.000 at L0 for anomaly**: a shallow-layer artefact; the CV-selected layer is L2 which has clean selectivity. If concerned, drop L0 from the reported per-layer curve.
- **MLP arch (128 ReLU in code vs 512 tanh in design doc)**: paper only says "one-hidden-layer MLP" — no discrepancy in the paper itself. Update the design doc to match the code.
- **PCA k clamp / `parse_rate=1.0` hardcoded field**: these are cosmetic; correct the code comments but no rerun.
- **Probe splits done on FactoryBench test items**: fine as long as we don't train the probe on the same items that appear in the panel's L4 scoring. Confirmed OK from the ID lists.

---

## Ordering + expected wall clock

1. Fixes 1, 3, 5, 6 are pure code — apply first, then rerun the full pipeline once (~2 hrs on your GPU).
2. Fixes 2, 4, 7, 10 need targeted `--behavioural-only` or `--randomize-weights` reruns — ~30 min each.
3. Fix 8 (multi-seed) — triple the wall clock of step 1.
4. Fix 9 (truncation) — verify cache invalidation, may re-extract activations (~1 hr).

**Total realistic budget: half a day of engineer + one full night of GPU.**

Once all reruns are done, ping me with the new `results.json` and `summary_table.csv` and I'll update the four paper numbers in `_paper_body.tex` §`app:probing` in one commit.
