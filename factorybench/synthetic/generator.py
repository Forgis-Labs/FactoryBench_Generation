"""
Synthetic Scenario Generator for FactoryBench.

Generates troubleshooting scenarios with full structural causal models (SCMs)
enabling evaluation across all three rungs of Pearl's causal hierarchy.

Key capabilities:
- Configurable causal graph generation
- Structural equation assignment (linear, nonlinear, threshold)
- Fault injection with various fault types
- Counterfactual label generation
- Intervention simulation

Example:
    from factorybench.synthetic import ScenarioGenerator, ScenarioConfig, DifficultyTier

    # Generate from tier
    generator = ScenarioGenerator(seed=42)
    config = ScenarioConfig.from_tier(DifficultyTier.TECHNICIAN)
    scenario = generator.generate(config)

    # Generate dataset
    dataset = generator.generate_dataset(
        n_scenarios=100,
        tiers=[DifficultyTier.APPRENTICE, DifficultyTier.TECHNICIAN],
    )
"""

import random
import uuid
from typing import List, Dict, Callable, Optional, Tuple
import numpy as np

from .config import (
    ScenarioConfig,
    DifficultyTier,
    GraphType,
    EquationType,
    FaultType,
)
from .scenario import (
    Scenario,
    CausalGraph,
    CounterfactualLabel,
    InterventionOutcome,
)


class ScenarioGenerator:
    """
    Generator for synthetic troubleshooting scenarios.

    Creates scenarios with known ground-truth causal structures and labels
    for rigorous evaluation of RCA methods.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator with optional seed for reproducibility.

        Args:
            seed: Random seed for reproducible generation
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(self, config: ScenarioConfig) -> Scenario:
        """
        Generate a single troubleshooting scenario.

        Args:
            config: Scenario configuration

        Returns:
            Scenario with all labels for hierarchical evaluation
        """
        # Generate unique ID
        scenario_id = f"syn_{uuid.uuid4().hex[:8]}"

        # Step 1: Generate variable names
        variable_names = config.variable_names or self._generate_variable_names(
            config.n_variables
        )

        # Step 2: Generate causal graph
        causal_graph = self._generate_causal_graph(
            variable_names,
            config.graph_type,
            config.edge_density,
        )

        # Step 3: Assign structural equations
        structural_equations = self._assign_structural_equations(
            causal_graph,
            config.equation_type,
            config.coefficient_range,
        )

        # Step 4: Select root cause
        root_cause = self._select_root_cause(causal_graph)

        # Step 5: Determine fault onset time
        onset_min, onset_max = config.fault_onset_range
        fault_onset = int(config.n_timesteps * random.uniform(onset_min, onset_max))

        # Step 6: Generate normal operation data
        timestamps = np.arange(config.n_timesteps) / config.sampling_rate
        time_series_normal = self._simulate_normal(
            causal_graph,
            structural_equations,
            config.n_timesteps,
            config.noise_level,
        )

        # Step 7: Inject fault and propagate
        time_series, anomaly_labels, affected_vars = self._inject_fault(
            time_series_normal,
            causal_graph,
            structural_equations,
            root_cause,
            config.fault_type,
            config.severity,
            fault_onset,
            config.noise_level,
        )

        # Step 8: Generate counterfactual labels
        counterfactual_labels = self._generate_counterfactual_labels(
            time_series_normal,
            causal_graph,
            structural_equations,
            root_cause,
            fault_onset,
            config.fault_type,
            config.severity,
        )

        # Step 9: Link to remediation (placeholder - needs manual corpus)
        remediation_links = self._get_remediation_links(
            root_cause,
            config.fault_type,
        )

        # Compute difficulty score
        difficulty_score = self._compute_difficulty_score(
            config, causal_graph, len(affected_vars)
        )

        return Scenario(
            scenario_id=scenario_id,
            config=config,
            time_series=time_series,
            timestamps=timestamps,
            variable_names=variable_names,
            causal_graph=causal_graph,
            structural_equations=structural_equations,
            root_cause=root_cause,
            fault_type=config.fault_type.value,
            fault_onset=fault_onset,
            fault_severity=config.severity,
            anomaly_labels=anomaly_labels,
            affected_variables=affected_vars,
            counterfactual_labels=counterfactual_labels,
            remediation_procedure=None,  # To be linked later
            remediation_links=remediation_links,
            tier=config.tier.value if config.tier else None,
            difficulty_score=difficulty_score,
        )

    def generate_dataset(
        self,
        n_scenarios: int,
        tiers: Optional[List[DifficultyTier]] = None,
        tier_weights: Optional[List[float]] = None,
    ) -> List[Scenario]:
        """
        Generate a dataset of multiple scenarios.

        Args:
            n_scenarios: Number of scenarios to generate
            tiers: List of difficulty tiers to sample from
            tier_weights: Weights for sampling each tier (default: equal)

        Returns:
            List of generated Scenarios
        """
        if tiers is None:
            tiers = list(DifficultyTier)

        if tier_weights is None:
            tier_weights = [1.0 / len(tiers)] * len(tiers)

        scenarios = []
        for i in range(n_scenarios):
            # Sample tier
            tier = random.choices(tiers, weights=tier_weights)[0]

            # Generate config with unique seed
            config = ScenarioConfig.from_tier(
                tier,
                seed=(self.seed + i) if self.seed else None,
            )

            # Generate scenario
            scenario = self.generate(config)
            scenarios.append(scenario)

        return scenarios

    # ========================================================================
    # PRIVATE METHODS - Graph Generation
    # ========================================================================

    def _generate_variable_names(self, n: int) -> List[str]:
        """Generate realistic industrial variable names."""
        prefixes = [
            "motor", "pump", "valve", "sensor", "temp", "pressure",
            "flow", "speed", "torque", "current", "vibration", "level",
        ]
        suffixes = ["_1", "_2", "_3", "_A", "_B", "_inlet", "_outlet", "_main"]

        names = []
        for i in range(n):
            prefix = prefixes[i % len(prefixes)]
            suffix = suffixes[i // len(prefixes) % len(suffixes)]
            names.append(f"{prefix}{suffix}")

        return names[:n]

    def _generate_causal_graph(
        self,
        nodes: List[str],
        graph_type: GraphType,
        edge_density: float,
    ) -> CausalGraph:
        """Generate causal graph structure."""
        n = len(nodes)
        edges = []

        if graph_type == GraphType.CHAIN:
            # Simple chain: 0 → 1 → 2 → ... → n-1
            for i in range(n - 1):
                edges.append((nodes[i], nodes[i + 1]))

        elif graph_type == GraphType.TREE:
            # Binary tree structure
            for i in range(1, n):
                parent_idx = (i - 1) // 2
                edges.append((nodes[parent_idx], nodes[i]))

        elif graph_type == GraphType.DAG:
            # Random DAG: for each pair (i, j) with i < j,
            # add edge with probability edge_density
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < edge_density:
                        edges.append((nodes[i], nodes[j]))

            # Ensure graph is connected (at least a spanning tree)
            connected = {nodes[0]}
            for i in range(1, n):
                if nodes[i] not in connected:
                    # Connect to a random connected node
                    parent = random.choice(list(connected))
                    edges.append((parent, nodes[i]))
                connected.add(nodes[i])

        elif graph_type == GraphType.CYCLIC:
            # DAG with one feedback edge
            for i in range(n - 1):
                edges.append((nodes[i], nodes[i + 1]))
            # Add feedback from last to first child
            if n > 2:
                edges.append((nodes[-1], nodes[1]))

        return CausalGraph(nodes=nodes, edges=edges)

    def _assign_structural_equations(
        self,
        graph: CausalGraph,
        equation_type: EquationType,
        coefficient_range: Tuple[float, float],
    ) -> Dict[str, Callable]:
        """Assign structural equations to each variable."""
        equations = {}
        coef_min, coef_max = coefficient_range

        for node in graph.nodes:
            parents = graph.parents.get(node, [])

            if not parents:
                # Root node: just noise
                equations[node] = lambda noise, p=parents: noise
            else:
                # Generate coefficients for parents
                coefficients = {
                    p: random.uniform(coef_min, coef_max) for p in parents
                }

                if equation_type == EquationType.LINEAR:
                    def eq(parent_vals, noise, c=coefficients):
                        return sum(c[p] * parent_vals.get(p, 0) for p in c) + noise
                    equations[node] = eq

                elif equation_type == EquationType.NONLINEAR:
                    def eq(parent_vals, noise, c=coefficients):
                        linear = sum(c[p] * parent_vals.get(p, 0) for p in c)
                        return np.tanh(linear) * 2 + noise
                    equations[node] = eq

                elif equation_type == EquationType.THRESHOLD:
                    threshold = random.uniform(0.5, 1.5)
                    def eq(parent_vals, noise, c=coefficients, t=threshold):
                        linear = sum(c[p] * parent_vals.get(p, 0) for p in c)
                        return float(linear > t) + noise
                    equations[node] = eq

                elif equation_type in (EquationType.MIXED, EquationType.SWITCHING):
                    # Randomly choose between linear and nonlinear
                    if random.random() < 0.5:
                        def eq(parent_vals, noise, c=coefficients):
                            return sum(c[p] * parent_vals.get(p, 0) for p in c) + noise
                    else:
                        def eq(parent_vals, noise, c=coefficients):
                            linear = sum(c[p] * parent_vals.get(p, 0) for p in c)
                            return np.tanh(linear) * 2 + noise
                    equations[node] = eq

        return equations

    def _select_root_cause(self, graph: CausalGraph) -> str:
        """Select a node to be the root cause of the fault."""
        # Prefer nodes that have descendants (so fault can propagate)
        candidates = []
        for node in graph.nodes:
            descendants = graph.get_descendants(node)
            if len(descendants) > 0:
                candidates.append((node, len(descendants)))

        if not candidates:
            # Fall back to any node
            return random.choice(graph.nodes)

        # Weight by number of descendants
        total = sum(d for _, d in candidates)
        weights = [d / total for _, d in candidates]
        return random.choices([n for n, _ in candidates], weights=weights)[0]

    # ========================================================================
    # PRIVATE METHODS - Simulation
    # ========================================================================

    def _simulate_normal(
        self,
        graph: CausalGraph,
        equations: Dict[str, Callable],
        n_timesteps: int,
        noise_level: float,
    ) -> np.ndarray:
        """Simulate normal operation (no faults)."""
        n_vars = len(graph.nodes)
        time_series = np.zeros((n_timesteps, n_vars))
        node_to_idx = {node: i for i, node in enumerate(graph.nodes)}

        # Get topological order for simulation
        topo_order = graph.topological_sort()

        for t in range(n_timesteps):
            parent_vals = {}

            for node in topo_order:
                idx = node_to_idx[node]
                noise = np.random.normal(0, noise_level)

                # Get parent values from current timestep
                for p in graph.parents.get(node, []):
                    p_idx = node_to_idx[p]
                    parent_vals[p] = time_series[t, p_idx]

                # Compute value using structural equation
                eq = equations[node]
                time_series[t, idx] = eq(parent_vals, noise)

        return time_series

    def _inject_fault(
        self,
        time_series_normal: np.ndarray,
        graph: CausalGraph,
        equations: Dict[str, Callable],
        root_cause: str,
        fault_type: FaultType,
        severity: float,
        fault_onset: int,
        noise_level: float,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Inject fault at root cause and propagate through graph."""
        time_series = time_series_normal.copy()
        n_timesteps, n_vars = time_series.shape
        node_to_idx = {node: i for i, node in enumerate(graph.nodes)}
        root_idx = node_to_idx[root_cause]

        # Generate anomaly labels
        anomaly_labels = np.zeros(n_timesteps, dtype=int)
        anomaly_labels[fault_onset:] = 1

        # Get variables affected by fault (root cause and descendants)
        affected = {root_cause} | graph.get_descendants(root_cause)
        affected_vars = list(affected)

        # Apply fault to root cause
        for t in range(fault_onset, n_timesteps):
            t_rel = t - fault_onset  # Relative time since fault

            if fault_type == FaultType.BIAS_SUDDEN:
                time_series[t, root_idx] += severity

            elif fault_type == FaultType.BIAS_GRADUAL:
                # Ramp up over 100 steps
                ramp = min(t_rel / 100, 1.0)
                time_series[t, root_idx] += severity * ramp

            elif fault_type == FaultType.SPIKE:
                # Spike at onset, then decay
                if t_rel < 10:
                    time_series[t, root_idx] += severity * (1 - t_rel / 10)

            elif fault_type == FaultType.NOISE_INCREASE:
                time_series[t, root_idx] += np.random.normal(0, severity)

            elif fault_type == FaultType.OSCILLATION:
                time_series[t, root_idx] += severity * np.sin(t_rel * 0.5)

            elif fault_type == FaultType.STUCK:
                time_series[t, root_idx] = time_series[fault_onset - 1, root_idx]

            elif fault_type == FaultType.DROPOUT:
                time_series[t, root_idx] = np.nan

            elif fault_type == FaultType.INTERMITTENT:
                if (t_rel // 20) % 2 == 0:
                    time_series[t, root_idx] += severity

            elif fault_type == FaultType.DELAY:
                if t_rel >= 10:
                    time_series[t, root_idx] = time_series[t - 10, root_idx]

        # Propagate fault through causal graph
        topo_order = graph.topological_sort()
        for t in range(fault_onset, n_timesteps):
            parent_vals = {}

            for node in topo_order:
                if node == root_cause:
                    # Already modified
                    parent_vals[node] = time_series[t, node_to_idx[node]]
                    continue

                idx = node_to_idx[node]

                # Get parent values
                for p in graph.parents.get(node, []):
                    p_idx = node_to_idx[p]
                    parent_vals[p] = time_series[t, p_idx]

                # Recompute if any parent is affected
                if any(p in affected for p in graph.parents.get(node, [])):
                    noise = np.random.normal(0, noise_level)
                    eq = equations[node]
                    time_series[t, idx] = eq(parent_vals, noise)

                parent_vals[node] = time_series[t, idx]

        return time_series, anomaly_labels, affected_vars

    def _generate_counterfactual_labels(
        self,
        time_series_normal: np.ndarray,
        graph: CausalGraph,
        equations: Dict[str, Callable],
        root_cause: str,
        fault_onset: int,
        fault_type: FaultType,
        severity: float,
    ) -> List[CounterfactualLabel]:
        """Generate counterfactual query labels."""
        labels = []

        # Query 1: Would anomaly have occurred if root cause was normal?
        labels.append(CounterfactualLabel(
            query=f"Would the anomaly have occurred if {root_cause} had remained normal?",
            variable=root_cause,
            counterfactual_value=0.0,  # Normal value
            outcome=False,  # No anomaly
            probability_necessity=1.0,  # Certain prevention
        ))

        # Query 2: For each ancestor, would intervention have helped?
        for ancestor in graph.get_ancestors(root_cause):
            # Simplification: ancestors can't fully prevent if fault is at root
            labels.append(CounterfactualLabel(
                query=f"Would the anomaly have occurred if {ancestor} had been different?",
                variable=ancestor,
                counterfactual_value=0.0,
                outcome=True,  # Anomaly still occurs
                probability_necessity=0.0,
            ))

        # Query 3: For non-ancestors, intervention has no effect
        non_ancestors = set(graph.nodes) - graph.get_ancestors(root_cause) - {root_cause}
        for var in list(non_ancestors)[:2]:  # Limit to 2 examples
            labels.append(CounterfactualLabel(
                query=f"Would the anomaly have occurred if {var} had been different?",
                variable=var,
                counterfactual_value=0.0,
                outcome=True,  # Anomaly still occurs
                probability_necessity=0.0,
            ))

        return labels

    def _get_remediation_links(
        self,
        root_cause: str,
        fault_type: FaultType,
    ) -> List[str]:
        """Get links to relevant manual sections (placeholder)."""
        # This would link to a real documentation corpus
        # For now, return placeholder IDs based on fault type
        base_id = f"manual_{root_cause}_{fault_type.value}"
        return [base_id, f"{base_id}_procedure", f"{base_id}_safety"]

    def _compute_difficulty_score(
        self,
        config: ScenarioConfig,
        graph: CausalGraph,
        n_affected: int,
    ) -> float:
        """Compute difficulty score for the scenario."""
        # Factors that increase difficulty:
        # - More variables
        # - Higher edge density
        # - Nonlinear equations
        # - More affected variables

        score = 0.0

        # Variable count (normalized to 0-1)
        score += min(config.n_variables / 50, 1.0) * 0.25

        # Edge density
        score += config.edge_density * 0.25

        # Equation complexity
        if config.equation_type == EquationType.LINEAR:
            score += 0.1
        elif config.equation_type == EquationType.MIXED:
            score += 0.15
        else:
            score += 0.25

        # Propagation extent
        score += min(n_affected / config.n_variables, 1.0) * 0.25

        return score
