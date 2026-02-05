"""
Scenario class representing a generated troubleshooting scenario.

Contains all components needed for hierarchical causal evaluation:
- Time series data
- Causal graph (ground truth)
- Structural equations (for intervention simulation)
- Labels for all task types
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np


@dataclass
class CausalGraph:
    """
    Represents a causal graph with nodes and directed edges.

    Provides utilities for graph queries needed in evaluation.
    """
    nodes: List[str]
    edges: List[Tuple[str, str]]  # List of (parent, child) tuples
    adjacency: Dict[str, List[str]] = field(default_factory=dict)  # parent -> children
    parents: Dict[str, List[str]] = field(default_factory=dict)    # child -> parents

    def __post_init__(self):
        """Build adjacency structures from edges."""
        self.adjacency = {node: [] for node in self.nodes}
        self.parents = {node: [] for node in self.nodes}

        for parent, child in self.edges:
            self.adjacency[parent].append(child)
            self.parents[child].append(parent)

    def get_ancestors(self, node: str) -> set:
        """Get all ancestors of a node (recursive parents)."""
        ancestors = set()
        to_visit = list(self.parents.get(node, []))
        while to_visit:
            current = to_visit.pop()
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(self.parents.get(current, []))
        return ancestors

    def get_descendants(self, node: str) -> set:
        """Get all descendants of a node (recursive children)."""
        descendants = set()
        to_visit = list(self.adjacency.get(node, []))
        while to_visit:
            current = to_visit.pop()
            if current not in descendants:
                descendants.add(current)
                to_visit.extend(self.adjacency.get(current, []))
        return descendants

    def get_roots(self) -> List[str]:
        """Get nodes with no parents (potential root causes)."""
        return [node for node in self.nodes if not self.parents.get(node)]

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order (parents before children)."""
        in_degree = {node: len(self.parents.get(node, [])) for node in self.nodes}
        queue = [node for node in self.nodes if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for child in self.adjacency.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return result

    def to_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Convert to adjacency matrix format."""
        n = len(self.nodes)
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        matrix = np.zeros((n, n))

        for parent, child in self.edges:
            i, j = node_to_idx[parent], node_to_idx[child]
            matrix[i, j] = 1

        return matrix, self.nodes

    def to_networkx(self):
        """Convert to NetworkX DiGraph (requires networkx installed)."""
        try:
            import networkx as nx
            G = nx.DiGraph()
            G.add_nodes_from(self.nodes)
            G.add_edges_from(self.edges)
            return G
        except ImportError:
            raise ImportError("NetworkX required: pip install networkx")


@dataclass
class InterventionOutcome:
    """Result of simulating an intervention on the SCM."""
    intervention_variable: str
    intervention_value: float
    from_step: int
    time_series_post: np.ndarray  # Time series after intervention
    anomaly_resolved: bool        # Did intervention fix the anomaly?
    steps_to_resolution: Optional[int] = None  # How many steps until resolved


@dataclass
class CounterfactualLabel:
    """Label for a counterfactual query."""
    query: str                    # Natural language query
    variable: str                 # Variable being queried
    counterfactual_value: float   # Hypothetical value
    outcome: bool                 # Would anomaly have occurred?
    probability_necessity: float  # P(no anomaly | do(var=cf_value))


@dataclass
class Scenario:
    """
    A complete troubleshooting scenario with all labels for evaluation.

    This is the main output of the ScenarioGenerator and contains everything
    needed to evaluate methods across all rungs of the causal hierarchy.

    Attributes:
        scenario_id: Unique identifier
        config: Generation configuration
        time_series: Multivariate time series (T x N)
        variable_names: Names of each variable
        causal_graph: Ground-truth causal structure
        structural_equations: Functions defining causal mechanisms
        root_cause: Variable that is the root cause of the anomaly
        fault_onset: Timestep when fault begins
        anomaly_labels: Binary labels for each timestep (0=normal, 1=anomalous)
        counterfactual_labels: Labels for counterfactual queries
        remediation_links: IDs of relevant manual sections
    """
    # Identification
    scenario_id: str
    config: Any  # ScenarioConfig

    # Time series data
    time_series: np.ndarray           # Shape: (T, N)
    timestamps: np.ndarray            # Shape: (T,)
    variable_names: List[str]

    # Causal structure
    causal_graph: CausalGraph
    structural_equations: Dict[str, Callable]  # var -> function(parents, noise)

    # Fault information
    root_cause: str                   # Variable name of root cause
    fault_type: str                   # Type of fault injected
    fault_onset: int                  # Timestep when fault begins
    fault_severity: float             # Magnitude of fault

    # Labels for evaluation
    anomaly_labels: np.ndarray        # Shape: (T,), binary
    affected_variables: List[str]     # Variables affected by fault propagation

    # Counterfactual labels
    counterfactual_labels: List[CounterfactualLabel] = field(default_factory=list)

    # Remediation information
    remediation_procedure: Optional[str] = None
    remediation_links: List[str] = field(default_factory=list)

    # Difficulty metadata
    tier: Optional[str] = None
    difficulty_score: Optional[float] = None

    def get_normal_data(self) -> np.ndarray:
        """Get time series before fault onset."""
        return self.time_series[:self.fault_onset]

    def get_anomalous_data(self) -> np.ndarray:
        """Get time series after fault onset."""
        return self.time_series[self.fault_onset:]

    def simulate_intervention(
        self,
        variable: str,
        value: float,
        from_step: int,
        n_steps: int = 100
    ) -> InterventionOutcome:
        """
        Simulate intervention do(variable=value) starting at from_step.

        This is the key capability for IRCA (Interventional RCA Protocol)
        evaluation - testing whether methods identify true causal mechanisms.

        Args:
            variable: Variable to intervene on
            value: Value to set
            from_step: Timestep to begin intervention
            n_steps: Number of steps to simulate after intervention

        Returns:
            InterventionOutcome with post-intervention time series and resolution status
        """
        # This is a placeholder - actual implementation requires
        # running the SCM forward with the intervention
        raise NotImplementedError(
            "Intervention simulation requires SCM execution. "
            "Implement in generator.py using structural_equations."
        )

    def compute_blame_scores(self) -> Dict[str, float]:
        """
        Compute causal responsibility scores for each variable.

        Based on Halpern-Pearl actual causality definitions.

        Returns:
            Dict mapping variable names to responsibility scores [0, 1]
        """
        # Placeholder - implement using counterfactual analysis
        scores = {}
        ancestors = self.causal_graph.get_ancestors(self.root_cause)

        # Root cause has highest responsibility
        scores[self.root_cause] = 1.0

        # Ancestors share some responsibility
        for i, anc in enumerate(ancestors):
            scores[anc] = 0.5 / (i + 1)  # Decreasing with distance

        # Other variables have zero responsibility
        for var in self.variable_names:
            if var not in scores:
                scores[var] = 0.0

        return scores

    def to_dict(self) -> dict:
        """Serialize to dictionary format for storage."""
        return {
            "scenario_id": self.scenario_id,
            "config": self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config),
            "time_series": self.time_series.tolist(),
            "timestamps": self.timestamps.tolist(),
            "variable_names": self.variable_names,
            "causal_graph": {
                "nodes": self.causal_graph.nodes,
                "edges": self.causal_graph.edges,
            },
            "root_cause": self.root_cause,
            "fault_type": self.fault_type,
            "fault_onset": self.fault_onset,
            "fault_severity": self.fault_severity,
            "anomaly_labels": self.anomaly_labels.tolist(),
            "affected_variables": self.affected_variables,
            "counterfactual_labels": [
                {
                    "query": cf.query,
                    "variable": cf.variable,
                    "counterfactual_value": cf.counterfactual_value,
                    "outcome": cf.outcome,
                    "probability_necessity": cf.probability_necessity,
                }
                for cf in self.counterfactual_labels
            ],
            "remediation_procedure": self.remediation_procedure,
            "remediation_links": self.remediation_links,
            "tier": self.tier,
            "difficulty_score": self.difficulty_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scenario":
        """Load from dictionary format."""
        # Reconstruct causal graph
        causal_graph = CausalGraph(
            nodes=data["causal_graph"]["nodes"],
            edges=[tuple(e) for e in data["causal_graph"]["edges"]],
        )

        # Reconstruct counterfactual labels
        cf_labels = [
            CounterfactualLabel(
                query=cf["query"],
                variable=cf["variable"],
                counterfactual_value=cf["counterfactual_value"],
                outcome=cf["outcome"],
                probability_necessity=cf["probability_necessity"],
            )
            for cf in data.get("counterfactual_labels", [])
        ]

        return cls(
            scenario_id=data["scenario_id"],
            config=data.get("config"),
            time_series=np.array(data["time_series"]),
            timestamps=np.array(data["timestamps"]),
            variable_names=data["variable_names"],
            causal_graph=causal_graph,
            structural_equations={},  # Cannot serialize functions
            root_cause=data["root_cause"],
            fault_type=data["fault_type"],
            fault_onset=data["fault_onset"],
            fault_severity=data["fault_severity"],
            anomaly_labels=np.array(data["anomaly_labels"]),
            affected_variables=data["affected_variables"],
            counterfactual_labels=cf_labels,
            remediation_procedure=data.get("remediation_procedure"),
            remediation_links=data.get("remediation_links", []),
            tier=data.get("tier"),
            difficulty_score=data.get("difficulty_score"),
        )
