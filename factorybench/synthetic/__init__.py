"""
FactoryBench Synthetic Data Generator

Generates troubleshooting scenarios with full structural causal models (SCMs)
for rigorous evaluation across Pearl's causal hierarchy.

Key components:
- ScenarioConfig: Configuration for scenario generation
- ScenarioGenerator: Main generator class
- Scenario: Generated scenario with all labels
"""

from .config import ScenarioConfig, FaultType, GraphType, EquationType, DifficultyTier
from .generator import ScenarioGenerator
from .scenario import Scenario, CausalGraph, InterventionOutcome

__all__ = [
    "ScenarioConfig",
    "ScenarioGenerator",
    "Scenario",
    "FaultType",
    "GraphType",
    "EquationType",
    "DifficultyTier",
    "CausalGraph",
    "InterventionOutcome",
]
