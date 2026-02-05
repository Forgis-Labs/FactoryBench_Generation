"""
FactoryBench Evaluation Framework

Hierarchical evaluation across Pearl's causal ladder:
- Rung 1: Associational (telemetry literacy, anomaly detection)
- Rung 2: Interventional (causal discovery, RCA)
- Rung 3: Counterfactual (blame attribution, prevention)
- Rung 4: Remediation (retrieval, action generation)

Key classes:
- HierarchicalEvaluator: Main evaluation orchestrator
- CausalDiscoveryMetrics: Metrics for causal graph evaluation
- RCAMetrics: Metrics for root cause analysis
- ConformalWrapper: Calibrated prediction sets
"""

from .metrics import (
    CausalDiscoveryMetrics,
    RCAMetrics,
    CounterfactualMetrics,
    RemediationMetrics,
    compute_map_at_k,
    compute_hit_at_k,
    compute_mrr,
    compute_graph_f1,
    compute_shd,
)

from .hierarchical import HierarchicalEvaluator, HierarchicalResults
from .irca import IRCAProtocol, IRCAResult
from .conformal import ConformalRCA, ConformalResult

__all__ = [
    # Main evaluator
    "HierarchicalEvaluator",
    "HierarchicalResults",

    # Metrics
    "CausalDiscoveryMetrics",
    "RCAMetrics",
    "CounterfactualMetrics",
    "RemediationMetrics",

    # Metric functions
    "compute_map_at_k",
    "compute_hit_at_k",
    "compute_mrr",
    "compute_graph_f1",
    "compute_shd",

    # Novel evaluation protocols
    "IRCAProtocol",
    "IRCAResult",
    "ConformalRCA",
    "ConformalResult",
]
