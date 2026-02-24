#!/usr/bin/env python3
"""
Generate Level 3 prompt JSON files from normalized episodes.

Outputs to datasets/questions/level3/prompts by default.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from generate_promtps import generate_level3_questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Level 3 prompts from normalized episodes")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/normalized_episodes/aursad"),
        help="Directory of normalized episode JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/questions/level3/prompts"),
        help="Output directory for generated prompt JSON files",
    )
    parser.add_argument("--min-len", type=int, default=32, help="Minimum subseries length")
    parser.add_argument("--max-len", type=int, default=64, help="Maximum subseries length")
    parser.add_argument(
        "--samples-per-episode",
        type=int,
        default=1,
        help="Number of prompts to sample per episode",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--csv",
        dest="export_csv",
        action="store_true",
        default=True,
        help="Export time series as CSV files (default: enabled)",
    )
    parser.add_argument(
        "--no-csv",
        dest="export_csv",
        action="store_false",
        help="Disable CSV export",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[3]
    phrases_path = Path(__file__).with_name("phrases_level3.json")
    root_causes_path = repo_root / "datasets" / "rca" / "root_causes.json"
    anomalies_path = repo_root / "datasets" / "rca" / "anomalies.json"
    machines_path = repo_root / "datasets" / "machines" / "machines.json"

    generate_level3_questions(
        input_dir=args.input,
        output_dir=args.output,
        phrases_path=phrases_path,
        root_causes_path=root_causes_path,
        anomalies_path=anomalies_path,
        machines_path=machines_path,
        min_len=args.min_len,
        max_len=args.max_len,
        samples_per_episode=args.samples_per_episode,
        seed=args.seed,
        test_mode=args.export_csv,
    )


if __name__ == "__main__":
    main()
