"""Balanced episode pairing for the two-robot comparison templates.

L1.3 and L2.8 present two streams and ask four independent yes/no questions:
different robots (A), different anomalous states (B), different tasks (C), and
same task at a different phase (D). Both templates drew their pairs uniformly,
which left the labels badly skewed:

    L1.3   P(A)=0.503  P(B)=0.000  P(C)=0.689  P(D)=0.298
    L2.8   P(A)=0.300  P(B)=0.953  P(C)=0.617  P(D)=0.294

B was degenerate in both directions for the same reason: L1 pairs two healthy
episodes so the anomalous states can never differ, while L2 always makes the
first stream faulty so they almost always do. The rest is pool shape, with
7,553 UR3 episodes against 1,302 KUKA. Guessing the per-position majority
scored 0.723 and 0.744 against a chance of 0.500, and emitting one fixed string
scored 35.5% and 32.8% against 6.25%.

This module picks pairs by target cell instead, so no proposition is guessable
from its prior.

D is not free. It can only hold when the tasks match, so P(D) <= P(not C).
Pinning both C and D at 0.5 therefore forces every same-task pair to differ in
phase, which makes D exactly "not C": a solver that answers C gets D for free.
An earlier version of this module did precisely that, and D was the negation of
C in 98.8% of L1.3 items and 98.3% of L2.8 items.

D now gives up its even marginal to stay a real question. C holds at 0.5, and
the same-task half splits evenly between differing and matching phases, so
P(D) is 0.25 and D still has to be read off the windows. A, B and C remain at
0.5. Twelve of the sixteen answer strings are reachable.

Supply is not a limit. Over the 8,855-episode pool the tightest cell still has
684,308 distinct ordered pairs, against the few hundred per cell a balanced
template needs.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# (different_robot, different_anomaly, different_task, different_phase).
#
# D can only hold when the tasks match, so (C true, D true) is impossible and
# three (C, D) combinations remain. Asking for C and D both at 0.5 forces the
# third out: every C-false item then has to carry D true, which makes D exactly
# "not C" and hands a solver that letter for free. Measured on the release, D
# was the negation of C in 98.8% of L1.3 items and 98.3% of L2.8 items.
#
# So D gives up its even marginal to stay informative. C keeps 0.5, and the
# C-false half splits evenly between D true and D false, leaving P(D) at 0.25
# but making D a real question: given C is false, D is still a coin flip that
# has to be read off the phases.
CELLS: Tuple[Tuple[bool, bool, bool, bool], ...] = (
    # C true -> D false, weighted twice so P(C) stays at 0.5
    (False, False, True, False), (False, False, True, False),
    (False, True, True, False), (False, True, True, False),
    (True, False, True, False), (True, False, True, False),
    (True, True, True, False), (True, True, True, False),
    # C false, D true: same task, different phase
    (False, False, False, True),
    (False, True, False, True),
    (True, False, False, True),
    (True, True, False, True),
    # C false, D false: same task, same phase
    (False, False, False, False),
    (False, True, False, False),
    (True, False, False, False),
    (True, True, False, False),
)


def anomaly_states_differ(fault_a: Any, fault_b: Any) -> bool:
    """Mirror of the template's own rule for proposition B.

    True when exactly one stream is anomalous, or both are but with different
    faults. Kept here so the sampler targets precisely what the scorer records.
    """
    try:
        fa = int(float(fault_a or 0))
        fb = int(float(fault_b or 0))
    except (TypeError, ValueError):
        return False
    if (fa == 0) != (fb == 0):
        return True
    return fa != 0 and fb != 0 and fa != fb


def cycle_targets(count: int, seed: Optional[int] = None) -> List[Tuple[bool, bool, bool]]:
    """A shuffled, evenly balanced list of target cells of length ``count``.

    Shuffled rather than round-robin so cell membership does not correlate with
    position in the output, which would reintroduce a shortcut for anything
    reading the files in order.
    """
    reps = (count + len(CELLS) - 1) // len(CELLS)
    targets = (list(CELLS) * reps)[:count]
    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(targets)
    return targets


# --------------------------------------------------------------------------
# direct cell sampling
#
# Rejection sampling drew the primary episode before it knew the target, so a
# cell needing a cross-robot partner often had none within the retry budget and
# the item was dropped. With 1,302 KUKA episodes against 7,553 UR3 that pushed
# P(different robots) down to 0.30 however many retries were allowed. Indexing
# episodes by their comparison attributes and picking the *group pair* first
# removes the dependence on what the primary happened to be.
# --------------------------------------------------------------------------
def build_index(episodes: Sequence[Dict[str, Any]]) -> Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]]:
    """Group episodes by (robot, task, fault)."""
    index: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = {}
    for ep in episodes:
        index.setdefault((ep.get("robot"), ep.get("task"), ep.get("fault")), []).append(ep)
    return index


def _group_pairs_for(index, target) -> List[Tuple[Tuple, Tuple]]:
    """Every pair of groups whose attributes land in ``target``.

    Only the first three entries of ``target`` are decided by group attributes.
    The fourth, whether the displayed windows sit in different phases, needs
    the rows and is checked by the caller.
    """
    keys = list(index)
    out = []
    for ka in keys:
        for kb in keys:
            if (ka[0] != kb[0]) != target[0]:
                continue
            if anomaly_states_differ(ka[2], kb[2]) != target[1]:
                continue
            if (ka[1] != kb[1]) != target[2]:
                continue
            if ka == kb and len(index[ka]) < 2:
                continue
            out.append((ka, kb))
    return out


def sample_pair(
    index,
    target,
    used: Optional[set] = None,
    phases_differ: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
    group_pairs_cache: Optional[Dict[Any, List]] = None,
    rng: Optional[random.Random] = None,
    max_tries: int = 200,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """An unused episode pair landing in ``target``, or None.

    Group pairs are chosen uniformly rather than in proportion to how many
    episodes they hold, so a cell is not dominated by whichever combination
    happens to be most common in the corpus. ``used`` records unordered pairs
    already emitted so no two items compare the same two episodes.
    """
    rng = rng or random
    if group_pairs_cache is not None:
        attr_key = tuple(target[:3])
        pairs = group_pairs_cache.get(attr_key)
        if pairs is None:
            pairs = _group_pairs_for(index, target)
            group_pairs_cache[attr_key] = pairs
    else:
        pairs = _group_pairs_for(index, target)
    if not pairs:
        return None
    want_same_task = not target[2]
    # target[3] says whether the two displayed windows must sit in different
    # phases. It is only meaningful for same-task pairs; a different-task pair
    # is never asked for a phase difference.
    want_diff_phase = bool(target[3]) if len(target) > 3 else want_same_task
    for _ in range(max_tries):
        ka, kb = pairs[rng.randrange(len(pairs))]
        ga, gb = index[ka], index[kb]
        a = ga[rng.randrange(len(ga))]
        b = gb[rng.randrange(len(gb))]
        if a.get("key") == b.get("key"):
            continue
        if used is not None:
            token = frozenset((a.get("key"), b.get("key")))
            if token in used:
                continue
        if want_same_task and phases_differ is not None:
            if phases_differ(a, b) != want_diff_phase:
                continue
        if used is not None:
            used.add(frozenset((a.get("key"), b.get("key"))))
        return a, b
    return None
