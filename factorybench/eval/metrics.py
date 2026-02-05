"""
Evaluation metrics for FactoryBench.

Organized by rung of Pearl's causal hierarchy:
- Rung 1: Associational metrics (anomaly detection, pattern recognition)
- Rung 2: Interventional metrics (causal discovery, RCA)
- Rung 3: Counterfactual metrics (attribution, prevention)
- Rung 4: Remediation metrics (retrieval, action accuracy)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import numpy as np


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def compute_map_at_k(
    predictions: List[List[str]],
    ground_truths: List[str],
    k: int = 3
) -> float:
    """
    Compute Mean Average Precision at K.

    Args:
        predictions: List of ranked predictions for each sample
        ground_truths: List of true labels (single root cause per sample)
        k: Cutoff rank

    Returns:
        MAP@K score in [0, 1]
    """
    if not predictions or not ground_truths:
        return 0.0

    aps = []
    for preds, gt in zip(predictions, ground_truths):
        # Truncate to k
        preds_k = preds[:k] if len(preds) >= k else preds

        if gt in preds_k:
            # Position (1-indexed) where ground truth appears
            rank = preds_k.index(gt) + 1
            ap = 1.0 / rank
        else:
            ap = 0.0

        aps.append(ap)

    return np.mean(aps)


def compute_hit_at_k(
    predictions: List[List[str]],
    ground_truths: List[str],
    k: int = 3
) -> float:
    """
    Compute Hit Rate at K (whether true label is in top K).

    Args:
        predictions: List of ranked predictions for each sample
        ground_truths: List of true labels
        k: Cutoff rank

    Returns:
        Hit@K score in [0, 1]
    """
    if not predictions or not ground_truths:
        return 0.0

    hits = []
    for preds, gt in zip(predictions, ground_truths):
        preds_k = preds[:k]
        hits.append(1.0 if gt in preds_k else 0.0)

    return np.mean(hits)


def compute_mrr(
    predictions: List[List[str]],
    ground_truths: List[str],
) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        predictions: List of ranked predictions for each sample
        ground_truths: List of true labels

    Returns:
        MRR score in [0, 1]
    """
    if not predictions or not ground_truths:
        return 0.0

    rrs = []
    for preds, gt in zip(predictions, ground_truths):
        if gt in preds:
            rank = preds.index(gt) + 1
            rrs.append(1.0 / rank)
        else:
            rrs.append(0.0)

    return np.mean(rrs)


def compute_graph_f1(
    predicted_edges: Set[Tuple[str, str]],
    true_edges: Set[Tuple[str, str]],
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for edge prediction.

    Args:
        predicted_edges: Set of predicted (parent, child) edges
        true_edges: Set of ground-truth edges

    Returns:
        Dict with precision, recall, f1 scores
    """
    if not predicted_edges and not true_edges:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not predicted_edges:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not true_edges:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(predicted_edges & true_edges)
    fp = len(predicted_edges - true_edges)
    fn = len(true_edges - predicted_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_shd(
    predicted_edges: Set[Tuple[str, str]],
    true_edges: Set[Tuple[str, str]],
) -> int:
    """
    Compute Structural Hamming Distance.

    SHD = # missing edges + # extra edges + # reversed edges

    Args:
        predicted_edges: Set of predicted edges
        true_edges: Set of ground-truth edges

    Returns:
        SHD (lower is better)
    """
    # Missing edges
    missing = len(true_edges - predicted_edges)

    # Extra edges (excluding reversed)
    extra = 0
    reversed_count = 0
    for edge in predicted_edges - true_edges:
        reversed_edge = (edge[1], edge[0])
        if reversed_edge in true_edges:
            reversed_count += 1
        else:
            extra += 1

    return missing + extra + reversed_count


def compute_orientation_accuracy(
    predicted_edges: Set[Tuple[str, str]],
    true_edges: Set[Tuple[str, str]],
) -> float:
    """
    Compute accuracy of edge orientations for correctly identified edges.

    Args:
        predicted_edges: Set of predicted edges
        true_edges: Set of ground-truth edges

    Returns:
        Orientation accuracy in [0, 1]
    """
    # Find edges that exist in both (regardless of direction)
    pred_undirected = {frozenset(e) for e in predicted_edges}
    true_undirected = {frozenset(e) for e in true_edges}

    common = pred_undirected & true_undirected
    if not common:
        return 0.0

    correct = 0
    for edge_set in common:
        # Check if direction matches
        edge_tuple = tuple(edge_set)
        if len(edge_tuple) == 2:
            e1 = (edge_tuple[0], edge_tuple[1])
            e2 = (edge_tuple[1], edge_tuple[0])
            if (e1 in predicted_edges and e1 in true_edges) or \
               (e2 in predicted_edges and e2 in true_edges):
                correct += 1

    return correct / len(common)


# =============================================================================
# METRIC DATACLASSES
# =============================================================================

@dataclass
class CausalDiscoveryMetrics:
    """Metrics for causal discovery evaluation (Rung 2)."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    shd: int = 0
    orientation_accuracy: float = 0.0
    n_predicted_edges: int = 0
    n_true_edges: int = 0
    runtime_seconds: float = 0.0

    @classmethod
    def compute(
        cls,
        predicted_edges: Set[Tuple[str, str]],
        true_edges: Set[Tuple[str, str]],
        runtime: float = 0.0,
    ) -> "CausalDiscoveryMetrics":
        """Compute all causal discovery metrics."""
        f1_metrics = compute_graph_f1(predicted_edges, true_edges)
        shd = compute_shd(predicted_edges, true_edges)
        orient_acc = compute_orientation_accuracy(predicted_edges, true_edges)

        return cls(
            precision=f1_metrics["precision"],
            recall=f1_metrics["recall"],
            f1=f1_metrics["f1"],
            shd=shd,
            orientation_accuracy=orient_acc,
            n_predicted_edges=len(predicted_edges),
            n_true_edges=len(true_edges),
            runtime_seconds=runtime,
        )

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "shd": self.shd,
            "orientation_accuracy": self.orientation_accuracy,
            "n_predicted_edges": self.n_predicted_edges,
            "n_true_edges": self.n_true_edges,
            "runtime_seconds": self.runtime_seconds,
        }


@dataclass
class RCAMetrics:
    """Metrics for root cause analysis evaluation (Rung 2)."""
    map_at_1: float = 0.0
    map_at_2: float = 0.0
    map_at_3: float = 0.0
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    mrr: float = 0.0
    n_samples: int = 0

    @classmethod
    def compute(
        cls,
        predictions: List[List[str]],
        ground_truths: List[str],
    ) -> "RCAMetrics":
        """Compute all RCA metrics."""
        return cls(
            map_at_1=compute_map_at_k(predictions, ground_truths, k=1),
            map_at_2=compute_map_at_k(predictions, ground_truths, k=2),
            map_at_3=compute_map_at_k(predictions, ground_truths, k=3),
            hit_at_1=compute_hit_at_k(predictions, ground_truths, k=1),
            hit_at_3=compute_hit_at_k(predictions, ground_truths, k=3),
            mrr=compute_mrr(predictions, ground_truths),
            n_samples=len(ground_truths),
        )

    def to_dict(self) -> dict:
        return {
            "map_at_1": self.map_at_1,
            "map_at_2": self.map_at_2,
            "map_at_3": self.map_at_3,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "mrr": self.mrr,
            "n_samples": self.n_samples,
        }


@dataclass
class CounterfactualMetrics:
    """Metrics for counterfactual evaluation (Rung 3)."""
    attribution_accuracy: float = 0.0      # Correct identification of CF root cause
    prevention_accuracy: float = 0.0       # Correct identification of preventive intervention
    probability_necessity_mae: float = 0.0  # Error in probability of necessity estimates
    blame_ranking_accuracy: float = 0.0    # Spearman correlation of blame rankings
    n_queries: int = 0

    @classmethod
    def compute(
        cls,
        cf_predictions: List[bool],       # Predicted: would anomaly occur?
        cf_ground_truths: List[bool],     # True: would anomaly occur?
        pn_predictions: Optional[List[float]] = None,   # Probability of necessity
        pn_ground_truths: Optional[List[float]] = None,
    ) -> "CounterfactualMetrics":
        """Compute counterfactual metrics."""
        # Attribution accuracy (binary)
        correct = sum(1 for p, g in zip(cf_predictions, cf_ground_truths) if p == g)
        attribution_acc = correct / len(cf_predictions) if cf_predictions else 0.0

        # Probability of necessity MAE
        pn_mae = 0.0
        if pn_predictions and pn_ground_truths:
            pn_mae = np.mean(np.abs(
                np.array(pn_predictions) - np.array(pn_ground_truths)
            ))

        return cls(
            attribution_accuracy=attribution_acc,
            prevention_accuracy=attribution_acc,  # Same for now
            probability_necessity_mae=pn_mae,
            blame_ranking_accuracy=0.0,  # TODO: implement
            n_queries=len(cf_predictions),
        )

    def to_dict(self) -> dict:
        return {
            "attribution_accuracy": self.attribution_accuracy,
            "prevention_accuracy": self.prevention_accuracy,
            "probability_necessity_mae": self.probability_necessity_mae,
            "blame_ranking_accuracy": self.blame_ranking_accuracy,
            "n_queries": self.n_queries,
        }


@dataclass
class RemediationMetrics:
    """Metrics for remediation evaluation (Rung 4)."""
    # Retrieval metrics
    retrieval_mrr: float = 0.0
    retrieval_recall_at_3: float = 0.0
    retrieval_ndcg: float = 0.0

    # Generation metrics
    action_accuracy: float = 0.0          # Steps match ground truth
    step_ordering_accuracy: float = 0.0   # Correct order of steps
    safety_compliance: float = 0.0        # No dangerous omissions
    hallucination_rate: float = 0.0       # Fabricated content

    # Groundedness
    citation_accuracy: float = 0.0        # Citations are valid
    groundedness_score: float = 0.0       # Content supported by retrieved docs

    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "retrieval_mrr": self.retrieval_mrr,
            "retrieval_recall_at_3": self.retrieval_recall_at_3,
            "retrieval_ndcg": self.retrieval_ndcg,
            "action_accuracy": self.action_accuracy,
            "step_ordering_accuracy": self.step_ordering_accuracy,
            "safety_compliance": self.safety_compliance,
            "hallucination_rate": self.hallucination_rate,
            "citation_accuracy": self.citation_accuracy,
            "groundedness_score": self.groundedness_score,
            "n_samples": self.n_samples,
        }


# =============================================================================
# CAUSAL SUFFICIENCY SCORE (CSS)
# =============================================================================

def compute_causal_sufficiency_score(
    cd_f1_scores: List[float],
    rca_map_scores: List[float],
    target_map: float = 0.7,
) -> float:
    """
    Compute Causal Sufficiency Score (CSS).

    CSS answers: "What CD F1 is needed to achieve target RCA MAP@3?"

    Based on empirical observation that MAP@3 ≈ α(1 - exp(-β·F1)) + γ

    Args:
        cd_f1_scores: List of causal discovery F1 scores
        rca_map_scores: Corresponding RCA MAP@3 scores
        target_map: Target MAP@3 to achieve

    Returns:
        CSS: Minimum F1 needed for target MAP@3
    """
    if len(cd_f1_scores) < 3:
        return 0.5  # Default if insufficient data

    # Fit exponential model
    from scipy.optimize import curve_fit

    def model(f1, alpha, beta, gamma):
        return alpha * (1 - np.exp(-beta * f1)) + gamma

    try:
        params, _ = curve_fit(
            model,
            cd_f1_scores,
            rca_map_scores,
            p0=[0.65, 3.2, 0.12],
            bounds=([0, 0, 0], [1, 10, 0.5]),
        )
        alpha, beta, gamma = params

        # Solve for F1 given target MAP
        # target = α(1 - exp(-β·F1)) + γ
        # (target - γ)/α = 1 - exp(-β·F1)
        # exp(-β·F1) = 1 - (target - γ)/α
        # F1 = -ln(1 - (target - γ)/α) / β

        inner = 1 - (target_map - gamma) / alpha
        if inner <= 0:
            return 1.0  # Target unachievable
        if inner >= 1:
            return 0.0  # Any F1 achieves target

        css = -np.log(inner) / beta
        return min(max(css, 0.0), 1.0)

    except Exception:
        # Fallback to linear interpolation
        sorted_pairs = sorted(zip(cd_f1_scores, rca_map_scores))
        for f1, map_score in sorted_pairs:
            if map_score >= target_map:
                return f1
        return 1.0  # Target not achieved
