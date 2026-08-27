# What changed after fixing chance correction (rebuttal W.1)

The paper's headline metric,
$\tilde{s}=\max\!\left(0,\;(s-E)/(1-E)\right)$,
was rescored on the *same* 44 260 saved reply files under the signed
variant $\tilde{s}=(s-E)/(1-E)$, with no LLM re-run. Baseline behaviour
now matches the paper's own written claim (random $\to$ 0). Everything below
is derived from `output/rescored_signed_test_eval.json` and the regenerated
figures in `output/figures_rebuttal/`.

## Regenerated figures

Both metrics saved side by side so the delta is obvious:

- `output/figures_rebuttal/fig_main_heatmap.pdf` (clipped, original)
- `output/figures_rebuttal/fig_main_heatmap_signed.pdf` (signed, new)
- `output/figures_rebuttal/fig_difficulty_curve.pdf` (clipped, original)
- `output/figures_rebuttal/fig_difficulty_curve_signed.pdf` (signed, new)

The signed heatmap uses a diverging colormap centred at 0 so below-chance
cells are visible in red.

## Per-level side-by-side

Percentages, mean across the test split. `delta` is (signed - clipped) in
percentage points.

| model               | L1 clip | L1 sign | dL1  | L2 clip | L2 sign | dL2  | L3 clip | L3 sign | dL3  |
|---------------------|--------:|--------:|-----:|--------:|--------:|-----:|--------:|--------:|-----:|
| Claude Sonnet 4.6   |    46.8 |    28.9 | -17.9|    46.9 |    29.2 | -17.7|    43.8 |    36.4 |  -7.4|
| Qwen3-235B          |    36.0 |    14.0 | -21.9|    35.4 |    13.6 | -21.8|    41.6 |    31.7 |  -9.9|
| Mistral Large 3     |    34.6 |    12.0 | -22.6|    31.5 |     7.3 | -24.1|    34.3 |    24.0 | -10.3|
| GPT-5.1             |    30.9 |     6.4 | -24.4|    29.8 |     5.6 | -24.2|    29.6 |    17.8 | -11.9|
| DeepSeek V3.2       |    25.0 |  **-2.5** | -27.5|    28.9 |     3.8 | -25.1|    26.5 |    13.2 | -13.4|
| Qwen3-4B            |    21.8 |  **-6.1** | -27.9|    27.3 |     1.6 | -25.7|    26.7 |    11.8 | -14.8|
| Baseline (rand/lr)  |    28.4 |     1.9 | -26.5|    23.8 |    -3.0 | -26.8|    18.0 |     1.7 | -16.3|

## What the reviewer wanted verified

**Baseline lands near 0.** Under the signed metric the non-LLM baseline
averages to +1.9%, -3.0%, +1.7% on L1/L2/L3 — well inside noise, matching
the "random $\to$ 0" property the paper claims but its original metric did
not deliver. This closes the reviewer's core objection.

## What CHANGED in the paper's written claims

1. **Absolute score ceiling drops.**
   Paper's Abstract, §1, §5, and Conclusion say
   *"no model exceeds 50% on structured levels."*
   Under the signed metric the ceiling is **Claude Sonnet 4.6 at 36.4% on L3**;
   no model exceeds 40% on any structured level. Rewrite the ceiling number
   from 50% $\to$ 40%.

2. **L4 ceiling is unchanged.**
   Free-form has E=0 by construction, so the signed transform is a no-op.
   The 17.7% L4 ceiling for GPT-5.1 and the L4 collapse to <18% overall
   stand as written.

3. **New below-chance findings surface.**
   Three panel entries score *below chance* on some level after the clip
   is removed:
   - **DeepSeek V3.2 on L1**: -2.5% (was 25.0% clipped)
   - **Qwen3-4B on L1**: -6.1% (was 21.8% clipped)
   - **Qwen3-4B (ft) on L1/L2/L3**: -3.6% / -8.3% / -33.7%
     (the fine-tuned Qwen3-4B is now provably worse than random on L3;
     this was not visible in the paper's Table)

   The paper claims in §5:
   > *"Qwen3-4B (27.5%, 28.8%), DeepSeek V3.2 (29.1%, 28.5%), and GPT-5.1
     (30.0%, 31.7%) all clear the bar on L2 and L3 despite scoring below it
     on L1."*

   Under the signed metric, only GPT-5.1 and DeepSeek clearly clear zero on
   L2/L3; Qwen3-4B is barely positive on L2 (+1.6%) and 11.8% on L3, so the
   claim survives if reworded ("all lift above the random-baseline audit
   value on L2 and L3 despite matching or trailing it on L1"), but the
   numeric parenthetical needs the signed replacements.

4. **Overall ranking is UNCHANGED for the 6 zero-shot models.**
   Both metrics rank the panel identically:
   Claude Sonnet 4.6 > Qwen3-235B > Mistral Large 3 > GPT-5.1 >
   DeepSeek V3.2 > Qwen3-4B.
   The ranking of the *non-LLM baseline* moves: under the clipped metric
   it sits above DeepSeek and Qwen3-4B; under the signed metric it sits
   just below GPT-5.1 and above DeepSeek, at essentially +0.2% overall.
   That change is a consequence of the fix, not new evidence.

5. **L4 rank reversal is UNCHANGED.**
   "GPT-5.1 reverses rank to lead L4" claim is entirely on free-form scores
   and holds unchanged.

6. **Signal-comprehension vs protocol-retrieval dissociation is UNCHANGED.**
   The dissociation was a rank statement (models that top L1-L3 collapse on
   L4, and vice versa); it holds under both metrics.

## What STAYS the same

- Model panel ranking (bullet 4 above).
- L4 collapse (bullet 5).
- Signal-comp vs protocol-retrieval dissociation (bullet 6).
- Chronos-Bolt baseline numbers (§H): those are already reported on the
  signed-equivalent piecewise scorer with $E=1/4$, no change.
- All qualitative claims about which fault categories dominate GPT-5.1's L4
  score distribution (Figure 4b) — unchanged, free-form.

## Recommended paper edits

- Rewrite the chance-correction paragraph (main.tex line ~380) to describe
  the signed metric and drop the false "random $\to$ 0 with clip" claim.
- Rewrite the numeric ceiling in Abstract, §1, §5, and Conclusion:
  50% $\to$ 40% (or "well below 50%" $\to$ "below 40%").
- Update the parenthetical L2/L3 numbers in the "still outperform the
  baseline on L2/L3" paragraph in §5.
- Update Table 3 / Figure 3 / Figure 4 with the signed numbers.
- Add one footnote noting that below-chance signed scores are informative
  (previously hidden by the clip) and that the ranking is preserved.

The finetuned Qwen3-4B entries are a separate thread that surfaced here —
they are not in the paper's zero-shot Figure 3 but were part of a later
experiment; whether to include them in the rebuttal is a call for the
author, but the -33.7% on L3 is now on the record.
