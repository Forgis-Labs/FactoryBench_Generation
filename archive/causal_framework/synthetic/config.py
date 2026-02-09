"""
Configuration classes for synthetic scenario generation.

This module defines the configurable parameters for generating
troubleshooting scenarios with known ground-truth causal structures.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class DifficultyTier(str, Enum):
    """Difficulty tiers aligned with expertise levels."""
    APPRENTICE = "apprentice"  # Tier 1: Entry-level, simple scenarios
    TECHNICIAN = "technician"  # Tier 2: Intermediate, realistic complexity
    EXPERT = "expert"          # Tier 3: Advanced, complex cascading faults


class GraphType(str, Enum):
    """Types of causal graph structures."""
    CHAIN = "chain"         # Linear causal chain: A → B → C → D
    TREE = "tree"           # Hierarchical tree structure
    DAG = "dag"             # General directed acyclic graph
    CYCLIC = "cyclic"       # Contains feedback loops (for advanced scenarios)


class EquationType(str, Enum):
    """Types of structural equations for causal mechanisms."""
    LINEAR = "linear"           # V_i = Σ β_ij * Pa_j + noise
    NONLINEAR = "nonlinear"     # V_i = tanh(Σ β_ij * Pa_j) + noise
    THRESHOLD = "threshold"     # V_i = step(Σ β_ij * Pa_j - θ) + noise
    SWITCHING = "switching"     # Different regimes based on conditions
    MIXED = "mixed"             # Combination of above


class FaultType(str, Enum):
    """Types of faults that can be injected."""
    # Value faults
    BIAS_SUDDEN = "bias_sudden"       # Sudden constant offset
    BIAS_GRADUAL = "bias_gradual"     # Gradually increasing offset (drift)
    SPIKE = "spike"                   # Transient spike

    # Variance faults
    NOISE_INCREASE = "noise_increase"  # Increased observation noise
    OSCILLATION = "oscillation"        # Periodic oscillation added

    # Temporal faults
    DELAY = "delay"                   # Delayed response
    INTERMITTENT = "intermittent"     # On/off fault pattern

    # Structural faults
    DROPOUT = "dropout"               # Sensor failure (missing values)
    STUCK = "stuck"                   # Frozen at constant value


# Default parameters for each difficulty tier
TIER_DEFAULTS = {
    DifficultyTier.APPRENTICE: {
        "n_variables": (5, 10),
        "edge_density": (0.10, 0.20),
        "graph_type": GraphType.CHAIN,
        "equation_type": EquationType.LINEAR,
        "fault_types": [FaultType.BIAS_SUDDEN, FaultType.SPIKE],
        "noise_level": (0.05, 0.10),
        "n_timesteps": (500, 1000),
    },
    DifficultyTier.TECHNICIAN: {
        "n_variables": (10, 20),
        "edge_density": (0.20, 0.30),
        "graph_type": GraphType.DAG,
        "equation_type": EquationType.MIXED,
        "fault_types": [FaultType.BIAS_GRADUAL, FaultType.NOISE_INCREASE, FaultType.DELAY],
        "noise_level": (0.08, 0.15),
        "n_timesteps": (1000, 2000),
    },
    DifficultyTier.EXPERT: {
        "n_variables": (20, 50),
        "edge_density": (0.30, 0.45),
        "graph_type": GraphType.DAG,
        "equation_type": EquationType.NONLINEAR,
        "fault_types": list(FaultType),  # All fault types
        "noise_level": (0.10, 0.20),
        "n_timesteps": (2000, 5000),
    },
}


@dataclass
class ScenarioConfig:
    """
    Configuration for generating a troubleshooting scenario.

    Can be specified explicitly or derived from a difficulty tier.

    Example:
        # From tier (recommended)
        config = ScenarioConfig.from_tier(DifficultyTier.TECHNICIAN)

        # Explicit configuration
        config = ScenarioConfig(
            n_variables=15,
            graph_type=GraphType.DAG,
            edge_density=0.25,
            equation_type=EquationType.MIXED,
            fault_type=FaultType.BIAS_GRADUAL,
            severity=1.5,
            noise_level=0.1,
            n_timesteps=2000,
        )
    """
    # Graph structure
    n_variables: int = 10
    graph_type: GraphType = GraphType.DAG
    edge_density: float = 0.25  # Probability of edge between valid pairs

    # Causal mechanisms
    equation_type: EquationType = EquationType.LINEAR
    coefficient_range: tuple = (-2.0, 2.0)  # Range for causal coefficients

    # Fault injection
    fault_type: FaultType = FaultType.BIAS_GRADUAL
    severity: float = 1.5  # Magnitude of fault relative to normal variation
    fault_onset_range: tuple = (0.3, 0.7)  # Relative position in time series

    # Observation noise
    noise_level: float = 0.1  # Standard deviation of observation noise

    # Time series
    n_timesteps: int = 1000
    sampling_rate: float = 1.0  # Hz

    # Propagation
    propagation_delay: int = 0  # Steps of delay in causal propagation

    # Metadata
    tier: Optional[DifficultyTier] = None
    variable_names: Optional[List[str]] = None  # Custom names, or auto-generated

    @classmethod
    def from_tier(
        cls,
        tier: DifficultyTier,
        seed: Optional[int] = None,
        **overrides
    ) -> "ScenarioConfig":
        """
        Create configuration from a difficulty tier with optional overrides.

        Args:
            tier: Difficulty tier (APPRENTICE, TECHNICIAN, EXPERT)
            seed: Random seed for sampling within ranges
            **overrides: Override any default parameter

        Returns:
            ScenarioConfig with tier-appropriate defaults
        """
        import random
        if seed is not None:
            random.seed(seed)

        defaults = TIER_DEFAULTS[tier]

        # Sample from ranges
        n_var_range = defaults["n_variables"]
        n_variables = random.randint(n_var_range[0], n_var_range[1])

        density_range = defaults["edge_density"]
        edge_density = random.uniform(density_range[0], density_range[1])

        noise_range = defaults["noise_level"]
        noise_level = random.uniform(noise_range[0], noise_range[1])

        timestep_range = defaults["n_timesteps"]
        n_timesteps = random.randint(timestep_range[0], timestep_range[1])

        fault_type = random.choice(defaults["fault_types"])

        config = cls(
            tier=tier,
            n_variables=n_variables,
            graph_type=defaults["graph_type"],
            edge_density=edge_density,
            equation_type=defaults["equation_type"],
            fault_type=fault_type,
            noise_level=noise_level,
            n_timesteps=n_timesteps,
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "n_variables": self.n_variables,
            "graph_type": self.graph_type.value,
            "edge_density": self.edge_density,
            "equation_type": self.equation_type.value,
            "coefficient_range": self.coefficient_range,
            "fault_type": self.fault_type.value,
            "severity": self.severity,
            "fault_onset_range": self.fault_onset_range,
            "noise_level": self.noise_level,
            "n_timesteps": self.n_timesteps,
            "sampling_rate": self.sampling_rate,
            "propagation_delay": self.propagation_delay,
            "tier": self.tier.value if self.tier else None,
            "variable_names": self.variable_names,
        }
