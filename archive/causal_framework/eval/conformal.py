"""
Conformal Prediction for RCA Calibration.

Provides calibrated prediction sets with coverage guarantees,
enabling safe deployment of RCA methods.

Key metrics:
- Marginal coverage: P(true ∈ prediction_set) ≥ 1-α
- Set size: Average size of prediction sets (smaller is better)
- Conditional coverage: Coverage stratified by difficulty

Reference: Based on conformal prediction literature and RobustUQ (2025)
"""

from dataclasses import dataclass
from typing import List, Any, Optional, Tuple
import numpy as np


@dataclass
class ConformalResult:
    """Results from conformal calibration analysis."""
    # Coverage metrics
    marginal_coverage: float          # Overall coverage
    conditional_coverage_tier1: float  # Coverage on easy scenarios
    conditional_coverage_tier2: float  # Coverage on medium scenarios
    conditional_coverage_tier3: float  # Coverage on hard scenarios

    # Efficiency metrics
    average_set_size: float           # Average |C(X)|
    set_size_std: float               # Std dev of set sizes
    empty_set_rate: float             # Fraction of empty prediction sets

    # Calibration quality
    calibration_error: float          # |coverage - target|
    is_calibrated: bool               # coverage ≥ target - ε

    # Deployment readiness
    deployment_ready: bool            # Meets all criteria
    failure_reasons: List[str]        # Why not deployment ready

    def summary(self) -> str:
        """Return human-readable summary."""
        status = "✅ READY" if self.deployment_ready else "❌ NOT READY"
        lines = [
            f"Conformal Calibration Analysis: {status}",
            f"  Marginal Coverage: {self.marginal_coverage:.3f}",
            f"  Average Set Size: {self.average_set_size:.2f}",
            f"  Conditional Coverage (Tier 3): {self.conditional_coverage_tier3:.3f}",
        ]
        if self.failure_reasons:
            lines.append("  Issues:")
            for reason in self.failure_reasons:
                lines.append(f"    - {reason}")
        return "\n".join(lines)


class ConformalRCA:
    """
    Conformal prediction wrapper for RCA methods.

    Transforms RCA predictions into calibrated prediction sets
    with coverage guarantees.

    Example:
        # Wrap any RCA method
        conformal_rca = ConformalRCA(base_method=my_rca_method)

        # Calibrate on held-out data
        conformal_rca.calibrate(calibration_scenarios)

        # Get calibrated prediction sets
        for scenario in test_scenarios:
            prediction_set = conformal_rca.predict_set(scenario)
            # prediction_set guaranteed to contain true root cause
            # with probability ≥ 1-α
    """

    def __init__(
        self,
        base_method: Any,
        alpha: float = 0.1,          # Target miscoverage rate (1-α = coverage)
        score_type: str = "rank",    # "rank" or "probability"
    ):
        """
        Initialize conformal wrapper.

        Args:
            base_method: Base RCA method with predict() returning ranked list
            alpha: Target miscoverage rate (default 0.1 = 90% coverage)
            score_type: Type of nonconformity score
        """
        self.base_method = base_method
        self.alpha = alpha
        self.score_type = score_type
        self.calibration_scores: List[float] = []
        self.threshold: Optional[float] = None

    def calibrate(
        self,
        scenarios: List[Any],
    ) -> float:
        """
        Calibrate threshold on held-out scenarios.

        Args:
            scenarios: Calibration scenarios with known root causes

        Returns:
            Calibrated threshold
        """
        scores = []

        for scenario in scenarios:
            # Get predictions from base method
            predictions = self.base_method.predict(
                scenario.time_series,
                scenario.fault_onset,
                getattr(scenario, 'causal_graph', None),
            )

            # Compute nonconformity score for true root cause
            score = self._compute_score(predictions, scenario.root_cause)
            scores.append(score)

        self.calibration_scores = scores

        # Compute threshold as (1-α) quantile
        n = len(scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.threshold = np.quantile(scores, min(q, 1.0))

        return self.threshold

    def predict_set(
        self,
        scenario: Any,
    ) -> List[str]:
        """
        Get calibrated prediction set.

        Args:
            scenario: Test scenario

        Returns:
            Prediction set with coverage guarantee
        """
        if self.threshold is None:
            raise RuntimeError("Must call calibrate() before predict_set()")

        # Get predictions
        predictions = self.base_method.predict(
            scenario.time_series,
            scenario.fault_onset,
            getattr(scenario, 'causal_graph', None),
        )

        # Include all predictions with score ≤ threshold
        prediction_set = []
        for i, pred in enumerate(predictions):
            score = self._compute_single_score(i, len(predictions))
            if score <= self.threshold:
                prediction_set.append(pred)

        return prediction_set

    def evaluate(
        self,
        scenarios: List[Any],
        tier_labels: Optional[List[int]] = None,
    ) -> ConformalResult:
        """
        Evaluate conformal calibration on test scenarios.

        Args:
            scenarios: Test scenarios
            tier_labels: Optional tier labels (1, 2, 3) for conditional coverage

        Returns:
            ConformalResult with coverage and efficiency metrics
        """
        coverages = []
        set_sizes = []
        tier_coverages = {1: [], 2: [], 3: []}

        for i, scenario in enumerate(scenarios):
            # Get prediction set
            pred_set = self.predict_set(scenario)

            # Check coverage
            covered = scenario.root_cause in pred_set
            coverages.append(covered)
            set_sizes.append(len(pred_set))

            # Conditional coverage by tier
            if tier_labels:
                tier = tier_labels[i]
                tier_coverages[tier].append(covered)
            elif hasattr(scenario, 'tier'):
                tier_map = {"apprentice": 1, "technician": 2, "expert": 3}
                tier = tier_map.get(scenario.tier, 2)
                tier_coverages[tier].append(covered)

        # Compute metrics
        marginal = np.mean(coverages)
        avg_size = np.mean(set_sizes)
        size_std = np.std(set_sizes)
        empty_rate = np.mean([s == 0 for s in set_sizes])

        cond_cov = {
            t: np.mean(c) if c else 0.0
            for t, c in tier_coverages.items()
        }

        calibration_error = abs(marginal - (1 - self.alpha))
        is_calibrated = marginal >= (1 - self.alpha - 0.02)  # 2% tolerance

        # Check deployment readiness
        failure_reasons = []
        if marginal < 0.95:
            failure_reasons.append(f"Marginal coverage {marginal:.3f} < 0.95")
        if avg_size > 3:
            failure_reasons.append(f"Average set size {avg_size:.2f} > 3")
        if cond_cov.get(3, 0) < 0.90:
            failure_reasons.append(f"Tier 3 coverage {cond_cov.get(3, 0):.3f} < 0.90")

        deployment_ready = len(failure_reasons) == 0

        return ConformalResult(
            marginal_coverage=marginal,
            conditional_coverage_tier1=cond_cov.get(1, 0.0),
            conditional_coverage_tier2=cond_cov.get(2, 0.0),
            conditional_coverage_tier3=cond_cov.get(3, 0.0),
            average_set_size=avg_size,
            set_size_std=size_std,
            empty_set_rate=empty_rate,
            calibration_error=calibration_error,
            is_calibrated=is_calibrated,
            deployment_ready=deployment_ready,
            failure_reasons=failure_reasons,
        )

    def _compute_score(
        self,
        predictions: List[str],
        ground_truth: str,
    ) -> float:
        """
        Compute nonconformity score for ground truth.

        Args:
            predictions: Ranked predictions
            ground_truth: True root cause

        Returns:
            Nonconformity score (lower = more confident)
        """
        if ground_truth not in predictions:
            return float('inf')

        rank = predictions.index(ground_truth)

        if self.score_type == "rank":
            # Score = rank (0 = most confident)
            return rank
        elif self.score_type == "probability":
            # Score = 1 - estimated probability
            # Assuming probability decreases with rank
            prob = 1.0 / (rank + 1)
            return 1 - prob
        else:
            return rank

    def _compute_single_score(
        self,
        rank: int,
        total: int,
    ) -> float:
        """Compute score for a single rank position."""
        if self.score_type == "rank":
            return rank
        elif self.score_type == "probability":
            prob = 1.0 / (rank + 1)
            return 1 - prob
        else:
            return rank
