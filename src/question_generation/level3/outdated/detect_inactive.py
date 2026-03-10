#!/usr/bin/env python3
"""
Detect inactive prompts by checking the count of numeric constant features.

Rule:
    INACTIVE = numeric_constant_count >= threshold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_prompt_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.json"))
    return [path]


def _get_constants(prompt: dict[str, Any]) -> dict[str, Any]:
    notes = prompt.get("notes", {}) or {}
    constants = notes.get("constant_features", {})
    return constants if isinstance(constants, dict) else {}


def _count_constant_feature_keys(constants: dict[str, Any]) -> int:
    count = 0
    for key in constants.keys():
        if key == "joint_modes":
            count += 4
        else:
            count += 1
    return count


def is_inactive_from_constants(constants: dict[str, Any], threshold: float) -> tuple[bool, dict[str, Any]]:
    numeric_count = _count_constant_feature_keys(constants)
    return numeric_count >= threshold, {
        "threshold": threshold,
        "numeric_constants": numeric_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect inactive prompts by constant feature count")
    parser.add_argument("--prompt", type=Path, required=True, help="Prompt JSON file or directory")
    parser.add_argument(
        "--threshold",
        type=int,
        default=55,
        help="Minimum numeric constant feature count to mark inactive",
    )

    args = parser.parse_args()

    results = []
    for path in _iter_prompt_files(args.prompt):
        prompt = load_json(path)
        constants = _get_constants(prompt)
        inactive, details = is_inactive_from_constants(constants, args.threshold)
        results.append({
            "file": str(path),
            "inactive": inactive,
            "details": details,
        })

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()