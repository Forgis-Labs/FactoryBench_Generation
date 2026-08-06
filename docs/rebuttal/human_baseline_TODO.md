# TODO: complete the human expert baseline

Tracks what is left to turn the current 60-item baseline into the one promised
in the paper and in the reviewer comment. The commitment made in
`docs/rebuttal/human_eval_comment.md` and in Appendix `app:human_baseline` is:

> at least 100 Q&A pairs with a minimum of 25 per level ... reporting the full
> per-level results in an updated appendix for the camera-ready version, and
> releasing the baseline itself: for every item, the expert's answer, its score,
> and the method used to reach it.

## Where it stands

| level | items now | target | gap |
|---|---|---|---|
| L1 | 13 | 25 | +12 |
| L2 | 37 | 25 | met |
| L3 | 10 | 25 | +15 |
| L4 | 0 | 25 | +25 |
| **total** | **60** | **100** | **+52** |

Current scores (raw, benchmark grader): overall 0.917, L1 0.962, L2 0.980,
L3 0.625. Per-item data in `output/final50_gcp/expert_scored_benchmark_scoring.json`
and `output/final50_gcp/expert_l3_scored.json`.

## Work items

1. **L4 is the priority.** It is the level carrying the paper's strongest claim
   and the one with no human number at all. L4 is free-form, so answers need the
   three-judge ensemble in `scripts/score_replies_batch.py`, not the deterministic
   grader used for L1--L3. Budget for that being slower per item than L1--L3.
2. **Broaden L1 (+12) and L3 (+15).** Sample from the released test split so the
   ceiling is measured on the same items the panel is scored on.
3. **Score with the benchmark's own grader.** Same acceptance bounds, exact-match
   and per-position partial credit the panel receives. No separate human rubric.
4. **Update the paper.** `tab:human_baseline` and the main-text paragraph in
   `_paper_body.tex` (`\subsection{Signal comprehension across Levels 1--3}`).
   Both currently say 60 items across L1--L3.
5. **Release the artifact.** Per item: expert answer, score, and solution method.

## Things to keep straight

- **Raw vs chance-corrected.** The baseline table is raw scores; Figure 3 is
  signed chance-corrected. The paper states this explicitly in both places. Keep
  that separation when the numbers are updated, or add a chance-corrected column
  computed with `src/evaluation/chance_correct.py`.
- **The expert does not lead at L3** (0.625 against DeepSeek's 0.675). If the
  extended L3 sample changes that, the paragraph in `app:human_baseline` and the
  main-text sentence about L3 headroom both need rewriting.
- **Annotator count is not claimed anywhere.** The text says "a robotics expert"
  and reports no inter-annotator agreement. If more annotators are added, that
  becomes reportable and should be.
- **Items answered before the dataset fixes.** The identification channel leak,
  the ranking context overlap and the segment/context feature mismatch were all
  repaired after this baseline was collected. Any newly sampled items will come
  from the corrected release, so note the version split if the old and new
  samples are pooled.
