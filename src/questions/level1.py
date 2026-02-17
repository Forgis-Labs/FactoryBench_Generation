"""
Level 1 question generator: State identification.

Generates questions of the form:
  "Has joint {i} moved between {T1} and {T2} (threshold of {eps} rad)?"

Ground-truth is computed deterministically from the normalized JSON produced
by the UR3e normalizer. If `feedback_pos_{i}` exists it is used; otherwise
`feedback_speed_{i}` is integrated over the time window to estimate displacement.

If required signals are missing, the ground-truth is set to null.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing import Union

logger = logging.getLogger(__name__)


def load_normalized_episode(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_phrases(path: Union[str, Path]) -> List[str]:
    p = Path(path)
    if not p.exists():
        # fallback to default single phrase
        return ["Has joint {axis} moved between {t1}ms and {t2}ms (threshold {eps_q} rad)?"]
    try:
        with open(p, "r", encoding="utf-8") as f:
            phrases = json.load(f)
        if isinstance(phrases, list) and all(isinstance(s, str) for s in phrases):
            return phrases
    except Exception:
        pass
    return ["Has joint {axis} moved between {t1}ms and {t2}ms (threshold {eps_q} rad)?"]


def _get_signal_names(axis: int) -> Tuple[str, str]:
    pos = f"feedback_pos_{axis}"
    speed = f"feedback_speed_{axis}"
    return pos, speed


def compute_displacement_from_speeds(rows: List[Dict[str, Any]], start: int, end: int, axis: int) -> Optional[float]:
    """Estimate displacement (rad) by integrating `feedback_speed_{axis}` between row indices start..end.

    Returns None if speed signal missing or timestamps invalid.
    """
    speed_key = f"feedback_speed_{axis}"
    ts_key = "timestamp_ms"

    # Ensure indices in range
    if start < 0 or end >= len(rows) or start >= end:
        return None

    # Check presence
    if any(speed_key not in r or r[speed_key] is None for r in rows[start : end + 1]):
        return None
    if any(ts_key not in r or r[ts_key] is None for r in rows[start : end + 1]):
        return None

    displacement = 0.0
    # Integrate using trapezoidal rule over sample intervals
    for i in range(start, end):
        v1 = float(rows[i][speed_key])
        v2 = float(rows[i + 1][speed_key])
        t1 = int(rows[i][ts_key]) / 1000.0
        t2 = int(rows[i + 1][ts_key]) / 1000.0
        dt = t2 - t1
        if dt <= 0:
            return None
        displacement += 0.5 * (v1 + v2) * dt
    return displacement


def compute_displacement_from_positions(rows: List[Dict[str, Any]], start: int, end: int, axis: int) -> Optional[float]:
    pos_key = f"feedback_pos_{axis}"
    if any(pos_key not in r or r[pos_key] is None for r in (rows[start], rows[end])):
        return None
    try:
        p1 = float(rows[start][pos_key])
        p2 = float(rows[end][pos_key])
        return abs(p2 - p1)
    except Exception:
        return None


def pick_time_window(rows: List[Dict[str, Any]], min_dt_ms: int, max_dt_ms: int) -> Tuple[int, int]:
    n = len(rows)
    if n < 2:
        raise ValueError("Not enough samples to pick a time window")
    # Build list of indices that have valid timestamps
    valid_idx = [i for i, r in enumerate(rows) if r.get("timestamp_ms") is not None]
    if len(valid_idx) < 2:
        # fallback to endpoints
        return 0, n - 1

    attempts = 0
    while attempts < 1000:
        i_pos = random.randint(0, len(valid_idx) - 2)
        j_pos = random.randint(i_pos + 1, len(valid_idx) - 1)
        i = valid_idx[i_pos]
        j = valid_idx[j_pos]
        dt = rows[j]["timestamp_ms"] - rows[i]["timestamp_ms"]
        if dt is None:
            attempts += 1
            continue
        if min_dt_ms <= dt <= max_dt_ms:
            return i, j
        attempts += 1

    # fallback: return first and last valid indices
    return valid_idx[0], valid_idx[-1]


def generate_level1_questions(
    episode_json: Path,
    out_json: Path,
    n_questions: int = 100,
    min_dt_ms: int = 100,
    max_dt_ms: int = 2000,
    eps_q: float = 1e-3,
    seed: Optional[int] = None,
    phrases_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Generate Level 1 questions for a single normalized episode.

    Args:
        episode_json: path to normalized episode JSON (list of rows)
        out_json: where to write generated questions (JSON array)
        n_questions: number of questions to generate
        min_dt_ms, max_dt_ms: time-window size bounds (ms)
        eps_q: threshold (rad) for determining "moved"
        seed: RNG seed
    Returns:
        List of question dicts written to `out_json`.
    """
    if seed is not None:
        random.seed(seed)

    rows = load_normalized_episode(episode_json)
    # load phrase templates
    phrases = load_phrases(phrases_file or Path(__file__).with_name("phrases_level1.json"))
    n_rows = len(rows)
    if n_rows == 0:
        raise ValueError("Episode JSON contains no rows")

    questions: List[Dict[str, Any]] = []

    for qid in range(1, n_questions + 1):
        axis = random.randint(0, 5)
        start_idx, end_idx = pick_time_window(rows, min_dt_ms, max_dt_ms)
        t1 = rows[start_idx]["timestamp_ms"]
        t2 = rows[end_idx]["timestamp_ms"]

        # Compute displacement using available signals
        disp_pos = compute_displacement_from_positions(rows, start_idx, end_idx, axis)
        if disp_pos is not None:
            displacement = disp_pos
            method = "position"
        else:
            disp_speed = compute_displacement_from_speeds(rows, start_idx, end_idx, axis)
            displacement = disp_speed
            method = "speed_integration"

        if displacement is None:
            ground_truth = None
        else:
            ground_truth = bool(abs(displacement) > eps_q)

        # render question text from a randomly chosen phrase template
        template = random.choice(phrases)
        try:
            text = template.format(axis=axis, t1=t1, t2=t2, eps_q=eps_q)
        except Exception:
            text = f"Has joint {axis} moved between {t1}ms and {t2}ms (threshold {eps_q} rad)?"

        q = {
            "id": f"L1-{episode_json.stem}-{qid:04d}",
            "question": {"text": text, "level": 1},
            "context": {"episode": str(episode_json), "time_window": [t1, t2], "joint": axis},
            "params": {"eps_q": eps_q, "method": method},
            "answer": {"ground_truth": ground_truth, "evidence": {"displacement_rad": displacement}},
            "provenance": "deterministic",
        }

        questions.append(q)

    # Write output file
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)

    logger.info(f"Wrote {len(questions)} Level 1 questions to {out_json}")
    return questions


if __name__ == "__main__":
    # lightweight CLI for quick runs
    import argparse

    parser = argparse.ArgumentParser(description="Generate Level 1 questions from normalized UR3e JSON.")
    parser.add_argument("--input", type=Path, required=True, help="Normalized episode JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output questions JSON file")
    parser.add_argument("--n", type=int, default=100, help="Number of questions to generate")
    parser.add_argument("--min-dt-ms", type=int, default=100, help="Minimum time window (ms)")
    parser.add_argument("--max-dt-ms", type=int, default=2000, help="Maximum time window (ms)")
    parser.add_argument("--eps-q", type=float, default=1e-3, help="Movement threshold in radians")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    generate_level1_questions(
        args.input,
        args.output,
        n_questions=args.n,
        min_dt_ms=args.min_dt_ms,
        max_dt_ms=args.max_dt_ms,
        eps_q=args.eps_q,
        seed=args.seed,
    )
