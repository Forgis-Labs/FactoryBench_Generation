"""
FactoryBench Evaluation Framework

Q&A evaluation across 5 levels of machine understanding:
- Level 1: State Identification
- Level 2: Anomaly Detection
- Level 3: Root Cause Analysis
- Level 4: Counterfactual Reasoning
- Level 5: Procedure + Prior

Current implementation:
- Stage 1 runner (telemetry literacy) - working

TODO:
- LLM-Match scoring for open-ended Q&A
- Per-level evaluation metrics
"""

from .runner import run_stage1_evaluation

__all__ = [
    "run_stage1_evaluation",
]
