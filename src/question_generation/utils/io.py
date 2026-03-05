"""
Shared I/O utilities for question generators across all levels.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_templates(path: Path) -> List[Dict[str, Any]]:
    return load_json(path)


def load_events(path: Path) -> List[Dict[str, Any]]:
    return load_json(path)


def load_root_causes(path: Path) -> Dict[int, Dict[str, Any]]:
    """Return {fault_id: root_cause_dict}."""
    causes = load_json(path)
    by_id: Dict[int, Dict[str, Any]] = {}
    for item in causes:
        fid = item.get("fault_id")
        if isinstance(fid, int):
            by_id[fid] = item
    return by_id
