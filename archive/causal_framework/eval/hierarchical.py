"""
Hierarchical Causal Evaluator for FactoryBench.

Orchestrates evaluation across all rungs of Pearl's causal hierarchy,
providing a unified interface for comprehensive method assessment.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Protocol
import time
import numpy as np

from .metrics import (
    CausalDiscoveryMetrics,
    RCAMetrics,
    CounterfactualMetrics,
    RemediationMetrics,
    compute_causal_sufficiency_score,
)


# =============================================================================
# PROTOCOLS (Interface Definitions)
# =============================================================================

class CausalDiscoveryMethod(Protocol):
    """Protocol for causal discovery methods."""

    def fit(self, time_series: np.ndarray, variable_names: List[str]) -> None:
        """Fit the model to time series data."""
        ...

    def get_edges(self) -> List[tuple]:
        """Return discovered edges as list of (parent, child) tuples."""
        ...


class RCAMethod(Protocol):
    """Protocol for root cause analysis methods."""

    def predict(
        self,
        time_series: np.ndarray,
        anomaly_onset: int,
        causal_graph: Optional[Any] = None,
    ) -> List[str]:
        """Return ranked list of predicted root causes."""
        ...


class CounterfactualMethod(Protocol):
    """Protocol for counterfactual reasoning methods."""

    def predict_counterfactual(
        self,
        time_series: np.ndarray,
        intervention_variable: str,
        intervention_value: float,
    ) -> bool:
        """Predict whether anomaly would occur under counterfactual."""
        ...


class RemediationMethod(Protocol):
    """Protocol for remediation methods."""

    def retrieve(self, fault_description: str) -> List[str]:
        """Retrieve relevant manual section IDs."""
        ...

    def generate_procedure(
        self,
        fault_description: str,
        retrieved_context: List[str],
    ) -> str:
        """Generate remediation procedure."""
        ...


# =============================================================================
# RESULTS DATACLASS
# =============================================================================

@dataclass
class HierarchicalResults:
    """
    Results from hierarchical evaluation across all rungs.

    Provides scores at each level of the causal hierarchy plus
    aggregate metrics and novel FactoryBench metrics.
    """
    # Per-rung scores (0-1, higher is better)
    rung1_score: float = 0.0  # Associational
    rung2_score: float = 0.0  # Interventional
    rung3_score: float = 0.0  # Counterfactual
    rung4_score: float = 0.0  # Remediation

    # Detailed metrics
    causal_discovery: Optional[CausalDiscoveryMetrics] = None
    rca: Optional[RCAMetrics] = None
    counterfactual: Optional[CounterfactualMetrics] = None
    remediation: Optional[RemediationMetrics] = None

    # Novel FactoryBench metrics
    irca_score: float = 0.0           # Interventional RCA Protocol
    css: float = 0.0                  # Causal Sufficiency Score
    conformal_coverage: float = 0.0   # Calibration coverage
    conformal_set_size: float = 0.0   # Average prediction set size

    # Aggregate
    overall_score: float = 0.0        # Weighted combination
    causal_gap: float = 0.0           # Rung1 - Rung3 (the "causal gap")

    # Metadata
    n_scenarios: int = 0
    total_runtime_seconds: float = 0.0

    def compute_overall(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute weighted overall score.

        Default weights emphasize Rung 2-3 as these are novel evaluations.
        """
        if weights is None:
            weights = {
                "rung1": 0.15,
                "rung2": 0.30,
                "rung3": 0.35,
                "rung4": 0.20,
            }

        self.overall_score = (
            weights["rung1"] * self.rung1_score +
            weights["rung2"] * self.rung2_score +
            weights["rung3"] * self.rung3_score +
            weights["rung4"] * self.rung4_score
        )

        self.causal_gap = self.rung1_score - self.rung3_score

        return self.overall_score

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "rung1_score": self.rung1_score,
            "rung2_score": self.rung2_score,
            "rung3_score": self.rung3_score,
            "rung4_score": self.rung4_score,
            "causal_discovery": self.causal_discovery.to_dict() if self.causal_discovery else None,
            "rca": self.rca.to_dict() if self.rca else None,
            "counterfactual": self.counterfactual.to_dict() if self.counterfactual else None,
            "remediation": self.remediation.to_dict() if self.remediation else None,
            "irca_score": self.irca_score,
            "css": self.css,
            "conformal_coverage": self.conformal_coverage,
            "conformal_set_size": self.conformal_set_size,
            "overall_score": self.overall_score,
            "causal_gap": self.causal_gap,
            "n_scenarios": self.n_scenarios,
            "total_runtime_seconds": self.total_runtime_seconds,
        }

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "=" * 60,
            "FACTORYBENCH HIERARCHICAL EVALUATION RESULTS",
            "=" * 60,
            "",
            "RUNG SCORES (Higher is better)",
            "-" * 40,
            f"  Rung 1 (Associational):   {self.rung1_score:.3f}",
            f"  Rung 2 (Interventional):  {self.rung2_score:.3f}",
            f"  Rung 3 (Counterfactual):  {self.rung3_score:.3f}",
            f"  Rung 4 (Remediation):     {self.rung4_score:.3f}",
            "",
            "NOVEL METRICS",
            "-" * 40,
            f"  IRCA Score:               {self.irca_score:.3f}",
            f"  Causal Sufficiency (CSS): {self.css:.3f}",
            f"  Conformal Coverage:       {self.conformal_coverage:.3f}",
            "",
            "AGGREGATE",
            "-" * 40,
            f"  Overall Score:            {self.overall_score:.3f}",
            f"  Causal Gap (R1-R3):       {self.causal_gap:.3f}",
            "",
            f"Scenarios: {self.n_scenarios} | Runtime: {self.total_runtime_seconds:.1f}s",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# HIERARCHICAL EVALUATOR
# =============================================================================

class HierarchicalEvaluator:
    """
    Main evaluator for FactoryBench hierarchical assessment.

    Orchestrates evaluation across all rungs of Pearl's causal hierarchy,
    computing standard metrics and novel FactoryBench metrics (CSS, IRCA).

    Example:
        evaluator = HierarchicalEvaluator()

        results = evaluator.evaluate(
            scenarios=test_scenarios,
            cd_method=pc_algorithm,
            rca_method=causal_prio_rca,
            cf_method=counterfactual_net,
            remediation_method=rag_pipeline,
        )

        print(results.summary())
    """

    def __init__(
        self,
        include_rung1: bool = True,
        include_rung2: bool = True,
        include_rung3: bool = True,
        include_rung4: bool = True,
        compute_irca: bool = True,
        compute_css: bool = True,
        compute_conformal: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize evaluator with configuration.

        Args:
            include_rung*: Whether to evaluate each rung
            compute_irca: Whether to run IRCA protocol
            compute_css: Whether to compute Causal Sufficiency Score
            compute_conformal: Whether to compute conformal calibration
            verbose: Print progress updates
        """
        self.include_rung1 = include_rung1
        self.include_rung2 = include_rung2
        self.include_rung3 = include_rung3
        self.include_rung4 = include_rung4
        self.compute_irca = compute_irca
        self.compute_css = compute_css
        self.compute_conformal = compute_conformal
        self.verbose = verbose

    def evaluate(
        self,
        scenarios: List[Any],  # List[Scenario]
        cd_method: Optional[CausalDiscoveryMethod] = None,
        rca_method: Optional[RCAMethod] = None,
        cf_method: Optional[CounterfactualMethod] = None,
        remediation_method: Optional[RemediationMethod] = None,
    ) -> HierarchicalResults:
        """
        Run full hierarchical evaluation.

        Args:
            scenarios: List of Scenario objects to evaluate on
            cd_method: Causal discovery method (for Rung 2)
            rca_method: Root cause analysis method (for Rung 2)
            cf_method: Counterfactual reasoning method (for Rung 3)
            remediation_method: Remediation method (for Rung 4)

        Returns:
            HierarchicalResults with all metrics
        """
        start_time = time.time()
        results = HierarchicalResults(n_scenarios=len(scenarios))

        if self.verbose:
            print(f"Evaluating on {len(scenarios)} scenarios...")

        # Rung 1: Associational (telemetry literacy, anomaly detection)
        if self.include_rung1:
            if self.verbose:
                print("  Evaluating Rung 1 (Associational)...")
            results.rung1_score = self._evaluate_rung1(scenarios)

        # Rung 2: Interventional (causal discovery + RCA)
        if self.include_rung2 and (cd_method or rca_method):
            if self.verbose:
                print("  Evaluating Rung 2 (Interventional)...")

            cd_metrics, rca_metrics, rung2_score = self._evaluate_rung2(
                scenarios, cd_method, rca_method
            )
            results.causal_discovery = cd_metrics
            results.rca = rca_metrics
            results.rung2_score = rung2_score

        # Rung 3: Counterfactual
        if self.include_rung3 and cf_method:
            if self.verbose:
                print("  Evaluating Rung 3 (Counterfactual)...")

            cf_metrics, rung3_score = self._evaluate_rung3(scenarios, cf_method)
            results.counterfactual = cf_metrics
            results.rung3_score = rung3_score

        # Rung 4: Remediation
        if self.include_rung4 and remediation_method:
            if self.verbose:
                print("  Evaluating Rung 4 (Remediation)...")

            rem_metrics, rung4_score = self._evaluate_rung4(
                scenarios, remediation_method
            )
            results.remediation = rem_metrics
            results.rung4_score = rung4_score

        # Novel metrics
        if self.compute_irca and rca_method:
            if self.verbose:
                print("  Computing IRCA Score...")
            results.irca_score = self._compute_irca(scenarios, rca_method)

        if self.compute_css and results.causal_discovery and results.rca:
            if self.verbose:
                print("  Computing CSS...")
            results.css = self._compute_css(results.causal_discovery, results.rca)

        # Compute overall
        results.total_runtime_seconds = time.time() - start_time
        results.compute_overall()

        if self.verbose:
            print(results.summary())

        return results

    # =========================================================================
    # PRIVATE EVALUATION METHODS
    # =========================================================================

    def _evaluate_rung1(self, scenarios: List[Any]) -> float:
        """
        Evaluate Rung 1 (Associational) tasks.

        Currently a placeholder - integrate with existing telemetry_literacy
        evaluation from Stage 1.
        """
        # TODO: Integrate with existing telemetry_literacy evaluation
        # For now, return placeholder
        return 0.80

    def _evaluate_rung2(
        self,
        scenarios: List[Any],
        cd_method: Optional[CausalDiscoveryMethod],
        rca_method: Optional[RCAMethod],
    ) -> tuple:
        """
        Evaluate Rung 2 (Interventional) tasks.

        Returns:
            (CausalDiscoveryMetrics, RCAMetrics, rung2_score)
        """
        cd_metrics = None
        rca_metrics = None

        # Causal Discovery evaluation
        if cd_method:
            all_predicted_edges = []
            all_true_edges = []
            total_cd_time = 0.0

            for scenario in scenarios:
                start = time.time()

                # Run causal discovery
                cd_method.fit(
                    scenario.time_series,
                    scenario.variable_names,
                )
                predicted_edges = set(cd_method.get_edges())

                total_cd_time += time.time() - start

                # Get ground truth edges
                true_edges = set(scenario.causal_graph.edges)

                all_predicted_edges.append(predicted_edges)
                all_true_edges.append(true_edges)

            # Aggregate metrics
            combined_pred = set().union(*all_predicted_edges) if all_predicted_edges else set()
            combined_true = set().union(*all_true_edges) if all_true_edges else set()

            cd_metrics = CausalDiscoveryMetrics.compute(
                combined_pred,
                combined_true,
                runtime=total_cd_time,
            )

        # RCA evaluation
        if rca_method:
            all_predictions = []
            all_ground_truths = []

            for scenario in scenarios:
                # Run RCA
                predictions = rca_method.predict(
                    scenario.time_series,
                    scenario.fault_onset,
                    scenario.causal_graph if hasattr(rca_method, 'uses_graph') else None,
                )
                all_predictions.append(predictions)
                all_ground_truths.append(scenario.root_cause)

            rca_metrics = RCAMetrics.compute(all_predictions, all_ground_truths)

        # Compute Rung 2 score (weighted combination of CD and RCA)
        cd_score = cd_metrics.f1 if cd_metrics else 0.0
        rca_score = rca_metrics.map_at_3 if rca_metrics else 0.0

        # Weight RCA more heavily as it's the end goal
        rung2_score = 0.3 * cd_score + 0.7 * rca_score

        return cd_metrics, rca_metrics, rung2_score

    def _evaluate_rung3(
        self,
        scenarios: List[Any],
        cf_method: CounterfactualMethod,
    ) -> tuple:
        """
        Evaluate Rung 3 (Counterfactual) tasks.

        Returns:
            (CounterfactualMetrics, rung3_score)
        """
        all_cf_predictions = []
        all_cf_ground_truths = []

        for scenario in scenarios:
            for cf_label in scenario.counterfactual_labels:
                # Get prediction
                prediction = cf_method.predict_counterfactual(
                    scenario.time_series,
                    cf_label.variable,
                    cf_label.counterfactual_value,
                )

                all_cf_predictions.append(prediction)
                all_cf_ground_truths.append(cf_label.outcome)

        cf_metrics = CounterfactualMetrics.compute(
            all_cf_predictions,
            all_cf_ground_truths,
        )

        return cf_metrics, cf_metrics.attribution_accuracy

    def _evaluate_rung4(
        self,
        scenarios: List[Any],
        remediation_method: RemediationMethod,
    ) -> tuple:
        """
        Evaluate Rung 4 (Remediation) tasks.

        Returns:
            (RemediationMetrics, rung4_score)
        """
        # Placeholder - implement retrieval and generation evaluation
        rem_metrics = RemediationMetrics(n_samples=len(scenarios))
        return rem_metrics, 0.0

    def _compute_irca(
        self,
        scenarios: List[Any],
        rca_method: RCAMethod,
    ) -> float:
        """
        Compute Interventional RCA Protocol score.

        Tests whether intervening on predicted root cause resolves anomaly.
        """
        successes = 0
        total = 0

        for scenario in scenarios:
            if not hasattr(scenario, 'simulate_intervention'):
                continue

            # Get RCA prediction
            predictions = rca_method.predict(
                scenario.time_series,
                scenario.fault_onset,
            )

            if not predictions:
                continue

            top_prediction = predictions[0]

            # Simulate intervention
            try:
                outcome = scenario.simulate_intervention(
                    variable=top_prediction,
                    value=0.0,  # Set to normal
                    from_step=scenario.fault_onset,
                )
                if outcome.anomaly_resolved:
                    successes += 1
            except NotImplementedError:
                # Intervention simulation not implemented
                pass

            total += 1

        return successes / total if total > 0 else 0.0

    def _compute_css(
        self,
        cd_metrics: CausalDiscoveryMetrics,
        rca_metrics: RCAMetrics,
    ) -> float:
        """Compute Causal Sufficiency Score."""
        # Simple version: based on observed F1 and MAP@3
        # Full version would use multiple data points to fit the curve
        return compute_causal_sufficiency_score(
            [cd_metrics.f1],
            [rca_metrics.map_at_3],
            target_map=0.7,
        )
