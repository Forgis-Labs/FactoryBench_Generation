"""Chance-correction for FactoryBench raw scores.

Maps a raw item score ``s`` in [0, 1] to a signed chance-corrected score
``s_tilde = (s - E) / (1 - E)`` where ``E`` is the expected raw score
under uniform random guessing for that item's answer format.

By linearity of expectation, ``E[s_tilde] = 0`` under uniform random
guessing for every format by construction; ``s_tilde = 1`` corresponds
to a perfect answer. The transform is signed: below-chance items produce
negative values, so a model that is worse than random is visibly
penalised in the aggregate mean rather than clipped to zero. The
per-format floor is ``s_tilde_min = -E / (1 - E)``, which ranges from
``-1`` for multi-select (``E = 1/2``) to ``-1/3`` for tensor / ranking /
four-way MCQ (``E = 1/4``).

An optional ``clip=True`` flag on ``chance_correct(...)`` restores the
historical max-clipped behaviour used in earlier drafts of the paper
(``max(0, (s - E) / (1 - E))``). That variant does not preserve the
random-baseline-equals-zero property once averaged, because it clips
below-chance mass to zero before averaging; it is kept only for
back-compatibility with older aggregators. New callers should leave
``clip=False`` (the module default), matching the metric the paper
describes.

Per-format chance levels:

  * single-select MCQ:  E = 1/k where k is the option count for the item
                        (the released set uses k=3 for L1 t6 / L2 t10 and
                        k=4 elsewhere).
  * multi-select MCQ:   E = 1/2  (each T/F slot matches gold w.p. 1/2 under
                        uniform guessing; expected fraction = 1/2 regardless
                        of length).
  * ranking:            E = 1/n where n is the permutation length. The
                        expected number of fixed points of a uniform random
                        permutation is exactly 1, so E = 1/n. Released set
                        uses n=4.
  * tensor:             E = 1/4 by construction; raw scoring uses the
                        three-level piecewise scorer (1 within m, 0.5 within
                        2m, 0 otherwise) with per-channel margin calibrated
                        to m_j = R_j/12, which yields E = 3m/R = 1/4. So
                        the same correction as single-select MCQ applies.
  * free-form:          NOT chance-corrected. The Level-4 rubric is already
                        on a {0, 0.5, 1} scale by construction and a model
                        emitting random text essentially never lands a
                        non-zero rubric score, so E = 0 and the correction
                        is a no-op. Free-form scores are returned unchanged.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional


def _option_count(question: Optional[Dict[str, Any]]) -> int:
    """Return the number of options for a single-select item, defaulting to 4."""
    if not isinstance(question, dict):
        return 4
    opts = question.get("options")
    if isinstance(opts, dict) and opts:
        return len(opts)
    if isinstance(opts, (list, tuple)) and opts:
        return len(opts)
    return 4


def _permutation_length(question: Optional[Dict[str, Any]]) -> int:
    """Return the gold permutation length for a ranking item, defaulting to 4."""
    if isinstance(question, dict):
        ans = question.get("answer")
        if isinstance(ans, str) and ans:
            return len(ans.strip())
    return 4


def expected_chance_score(
    answer_format: str,
    question: Optional[Dict[str, Any]] = None,
) -> float:
    """Return E for the given format/item. Free-form returns 0 (no-op)."""
    if answer_format == "multiple_choice_single_select":
        k = _option_count(question)
        return 1.0 / max(1, k)
    if answer_format == "multiple_choice_multi_select":
        return 0.5
    if answer_format == "ranking":
        n = _permutation_length(question)
        return 1.0 / max(1, n)
    if answer_format == "tensor":
        # Calibrated per-channel margin m_j = R_j/12 yields E = 3m/R = 1/4
        # under the three-level piecewise scorer.
        return 0.25
    if answer_format == "numerical":
        # Single-scalar tensor branch; same calibration target.
        return 0.25
    if answer_format == "free_form":
        return 0.0
    return 0.0


def chance_correct(
    raw_score: Optional[float],
    answer_format: str,
    question: Optional[Dict[str, Any]] = None,
    clip: bool = False,
) -> Optional[float]:
    """Map a raw item score to its chance-corrected counterpart.

    Two modes controlled by ``clip``:
      * ``clip=True``  (default): the historical FactoryBench metric,
        ``s_tilde = max(0, (s - E) / (1 - E))``, upper-clipped at 1. Because
        below-chance mass is floored to 0, the *expected* value of this metric
        under uniform random guessing is strictly greater than 0 for MCQ,
        ranking and tensor formats (e.g. ~0.25 for single-select MCQ with k=4).
      * ``clip=False`` (signed): ``s_tilde = (s - E) / (1 - E)`` with only the
        upper clip at 1. Below-chance items produce negative values, and by
        linearity the expected value under uniform random guessing is exactly
        0 for every format. This is the correct chance correction if you want
        the "random baseline = 0" property to hold.

    Returns ``None`` if ``raw_score`` is ``None`` (ungraded item). Returns the
    raw score unchanged for ``free_form`` (which has E = 0 by construction).
    """
    if raw_score is None:
        return None
    if answer_format == "free_form":
        return float(raw_score)
    E = expected_chance_score(answer_format, question)
    if E >= 1.0 or math.isnan(E):
        # Item is uninformative (chance saturates to 1); treat any score >= 1 as 1.
        return 1.0 if float(raw_score) >= 1.0 else 0.0
    s = float(raw_score)
    corrected = (s - E) / (1.0 - E)
    if clip and corrected < 0.0:
        return 0.0
    if corrected > 1.0:
        return 1.0
    return corrected
