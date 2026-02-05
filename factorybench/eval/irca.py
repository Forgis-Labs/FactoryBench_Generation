"""
Interventional RCA Protocol (IRCA) for FactoryBench.

IRCA tests whether RCA methods identify true causal mechanisms
by simulating interventions on predicted root causes.

Key insight: Methods exploiting spurious correlations will have
low IRCA scores even if they have high standard MAP@K scores.

Reference: Inspired by NDSS 2025 "Interventional Root Cause Analysis"
"""

from dataclasses import dataclass
from typing import List, Any, Optional
import numpy as np


@dataclass
class IRCAResult:
    """Result from IRCA protocol evaluation."""
    intervention_success_rate: float  # ISR: fraction where intervention resolved anomaly
    standard_map_at_3: float          # Standard MAP@3 for comparison
    irca_gap: float                   # MAP@3 - ISR (larger = more spurious correlation)
    n_scenarios: int
    n_successful_interventions: int
    n_failed_interventions: int

    def interpretation(self) -> str:
        """Provide interpretation of IRCA results."""
        if self.irca_gap < 0.1:
            return "EXCELLENT: Method shows genuine causal understanding"
        elif self.irca_gap < 0.2:
            return "GOOD: Method mostly identifies true causes"
        elif self.irca_gap < 0.3:
            return "MODERATE: Method partially exploits spurious correlations"
        else:
            return "POOR: Method heavily relies on spurious correlations"


class IRCAProtocol:
    """
    Interventional RCA Protocol evaluator.

    Tests causal understanding by simulating interventions:
    1. Get RCA prediction for root cause
    2. Simulate do(predicted_root_cause = normal_value)
    3. Check if anomaly resolves

    A method with true causal understanding will have high ISR
    because intervening on the true cause eliminates the effect.
    """

    def __init__(
        self,
        n_intervention_steps: int = 100,
        resolution_threshold: float = 0.1,
        verbose: bool = True,
    ):
        """
        Initialize IRCA protocol.

        Args:
            n_intervention_steps: Steps to simulate after intervention
            resolution_threshold: Threshold for anomaly resolution
            verbose: Print progress
        """
        self.n_intervention_steps = n_intervention_steps
        self.resolution_threshold = resolution_threshold
        self.verbose = verbose

    def evaluate(
        self,
        scenarios: List[Any],
        rca_method: Any,
        standard_map: Optional[float] = None,
    ) -> IRCAResult:
        """
        Run IRCA protocol on scenarios.

        Args:
            scenarios: List of Scenario objects with simulate_intervention capability
            rca_method: RCA method with predict() interface
            standard_map: Pre-computed standard MAP@3 (or computed here)

        Returns:
            IRCAResult with ISR and comparison metrics
        """
        successful = 0
        failed = 0
        predictions_for_map = []
        ground_truths = []

        for i, scenario in enumerate(scenarios):
            if self.verbose and i % 10 == 0:
                print(f"  IRCA: Processing scenario {i+1}/{len(scenarios)}")

            # Get RCA prediction
            try:
                predictions = rca_method.predict(
                    scenario.time_series,
                    scenario.fault_onset,
                    getattr(scenario, 'causal_graph', None),
                )
            except Exception as e:
                if self.verbose:
                    print(f"    Warning: RCA prediction failed: {e}")
                continue

            if not predictions:
                continue

            predictions_for_map.append(predictions)
            ground_truths.append(scenario.root_cause)

            top_prediction = predictions[0]

            # Simulate intervention
            try:
                resolved = self._simulate_intervention(
                    scenario,
                    top_prediction,
                )
                if resolved:
                    successful += 1
                else:
                    failed += 1
            except NotImplementedError:
                # Fall back to ground truth comparison
                if top_prediction == scenario.root_cause:
                    successful += 1
                else:
                    failed += 1

        # Compute metrics
        total = successful + failed
        isr = successful / total if total > 0 else 0.0

        # Compute standard MAP@3 if not provided
        if standard_map is None:
            from .metrics import compute_map_at_k
            standard_map = compute_map_at_k(predictions_for_map, ground_truths, k=3)

        return IRCAResult(
            intervention_success_rate=isr,
            standard_map_at_3=standard_map,
            irca_gap=standard_map - isr,
            n_scenarios=total,
            n_successful_interventions=successful,
            n_failed_interventions=failed,
        )

    def _simulate_intervention(
        self,
        scenario: Any,
        intervention_variable: str,
    ) -> bool:
        """
        Simulate intervention and check if anomaly resolves.

        Args:
            scenario: Scenario object
            intervention_variable: Variable to intervene on

        Returns:
            True if anomaly resolved after intervention
        """
        # Try to use scenario's simulation capability
        if hasattr(scenario, 'simulate_intervention'):
            outcome = scenario.simulate_intervention(
                variable=intervention_variable,
                value=0.0,  # Set to "normal"
                from_step=scenario.fault_onset,
                n_steps=self.n_intervention_steps,
            )
            return outcome.anomaly_resolved

        # Fallback: Check if prediction matches ground truth
        # (assumes correct prediction = successful intervention)
        return intervention_variable == scenario.root_cause

    def compute_expected_isr_from_map(
        self,
        map_at_3: float,
        n_variables: int = 10,
    ) -> float:
        """
        Compute expected ISR if method has perfect causal understanding.

        For a method with true causal understanding, ISR ≈ MAP@3
        because identifying the true root cause means intervention works.

        For a correlation-based method, ISR << MAP@3 because
        correlates don't have causal effect.

        Args:
            map_at_3: Standard MAP@3 score
            n_variables: Number of variables in system

        Returns:
            Expected ISR under perfect causal understanding
        """
        # Under perfect causal understanding:
        # If MAP@3 = 0.7, then 70% of top predictions are correct root causes
        # Intervening on correct root cause resolves anomaly
        # So expected ISR = MAP@3

        return map_at_3
