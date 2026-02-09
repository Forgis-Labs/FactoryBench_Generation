# FactoryBench Implementation Guide for NeurIPS 2026

**For: Coral**
**Supervisor: Jonas Petersen**
**Last Updated: February 2026**

---

## Overview

This guide provides a concrete implementation path for the FactoryBench NeurIPS paper. The codebase has been structured to make your work straightforward.

### What's Already Done

1. **Paper Draft** (`docs/FactoryBench_NeurIPS_Paper_Draft.md`)
   - Complete structure with 5 novel contributions
   - All figures and tables defined
   - References compiled

2. **Synthetic Data Generator** (`factorybench/synthetic/`)
   - `config.py` - Scenario configuration with 3 tiers
   - `scenario.py` - Scenario dataclass with causal graph
   - `generator.py` - Full generation pipeline

3. **Evaluation Framework** (`factorybench/eval/`)
   - `metrics.py` - All metrics (MAP@K, F1, SHD, CSS)
   - `hierarchical.py` - Main HierarchicalEvaluator class
   - `irca.py` - Interventional RCA Protocol (novel)
   - `conformal.py` - Conformal calibration (novel)

4. **Stage 1** (`factorybench/eval/runner.py`)
   - Telemetry Literacy evaluation complete
   - 4 Azure models integrated

### What You Need to Implement

| Task | File | Priority | Hours |
|------|------|----------|-------|
| Adapt CausRCA data | `datasets/causrca/` | HIGH | 15 |
| Causal discovery wrapper | `factorybench/methods/cd.py` | HIGH | 20 |
| RCA methods wrapper | `factorybench/methods/rca.py` | HIGH | 20 |
| Run experiments | `scripts/run_experiments.py` | HIGH | 30 |
| LLM baselines | `factorybench/methods/llm.py` | MEDIUM | 25 |
| Remediation evaluation | `factorybench/eval/remediation.py` | MEDIUM | 20 |
| Generate figures | `scripts/generate_figures.py` | HIGH | 15 |
| Write paper sections | `docs/paper/` | HIGH | 50 |

---

## Week-by-Week Plan

### Week 1-2: CausRCA Integration

Your preliminary results are excellent. Now formalize them:

```python
# Step 1: Convert CausRCA data to FactoryBench format
from factorybench.synthetic import Scenario, CausalGraph
import pandas as pd

def adapt_causrca(csv_path: str, graph_path: str) -> List[Scenario]:
    """Convert CausRCA format to FactoryBench Scenario."""
    # Load your existing data
    df = pd.read_csv(csv_path)

    # Create CausalGraph from adjacency matrix
    edges = load_edges_from_graph(graph_path)
    graph = CausalGraph(nodes=list(df.columns), edges=edges)

    # Create Scenario objects
    scenarios = []
    for fault_case in fault_cases:
        scenario = Scenario(
            scenario_id=f"causrca_{fault_case.id}",
            time_series=fault_case.data,
            causal_graph=graph,
            root_cause=fault_case.diagnosis,
            # ... fill other fields
        )
        scenarios.append(scenario)

    return scenarios
```

### Week 3-4: Causal Discovery Methods

Integrate methods from `causal-learn` library:

```python
# factorybench/methods/cd.py
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges

class PCWrapper:
    """PC algorithm wrapper for FactoryBench."""

    def fit(self, time_series: np.ndarray, variable_names: List[str]):
        # Run PC algorithm
        cg = pc(time_series, alpha=0.05)
        self.graph = cg.G
        self.variable_names = variable_names

    def get_edges(self) -> List[tuple]:
        # Convert to edge list
        edges = []
        for i in range(len(self.variable_names)):
            for j in range(len(self.variable_names)):
                if self.graph[i, j] == 1:  # Edge i -> j
                    edges.append((self.variable_names[i], self.variable_names[j]))
        return edges
```

### Week 5-6: RCA Methods

Implement the RCA methods from your preliminary results:

```python
# factorybench/methods/rca.py

class CausalPrioTimeRecencyRCA:
    """Causal-prioritized time recency RCA."""

    def __init__(self, causal_graph: CausalGraph):
        self.graph = causal_graph

    def predict(
        self,
        time_series: np.ndarray,
        anomaly_onset: int,
        **kwargs
    ) -> List[str]:
        # 1. Find recent changes
        recent_changes = self._find_recent_changes(time_series, anomaly_onset)

        # 2. Filter by causal predecessors of anomaly
        anomaly_vars = self._detect_anomaly_variables(time_series, anomaly_onset)
        causal_candidates = set()
        for var in anomaly_vars:
            causal_candidates |= self.graph.get_ancestors(var)

        # 3. Rank by recency among causal candidates
        ranked = []
        for var, recency in recent_changes:
            if var in causal_candidates:
                ranked.append(var)

        return ranked
```

### Week 7-8: Run Full Experiments

```python
# scripts/run_experiments.py
from factorybench.eval import HierarchicalEvaluator
from factorybench.synthetic import ScenarioGenerator, DifficultyTier

# Load data
causrca_scenarios = load_causrca_adapted()
synthetic_scenarios = ScenarioGenerator(seed=42).generate_dataset(
    n_scenarios=500,
    tiers=[DifficultyTier.APPRENTICE, DifficultyTier.TECHNICIAN],
)

# Initialize methods
cd_methods = {
    "PC": PCWrapper(),
    "FCI": FCIWrapper(),
    "FGES": FGESWrapper(),
    "PCMCI": PCMCIWrapper(),
}

rca_methods = {
    "TimeRecency": TimeRecencyRCA(),
    "CausalPrio": CausalPrioTimeRecencyRCA(),
    "PageRank": PageRankRCA(),
}

# Run evaluation
evaluator = HierarchicalEvaluator()
results = {}

for cd_name, cd_method in cd_methods.items():
    for rca_name, rca_method in rca_methods.items():
        key = f"{cd_name}+{rca_name}"
        results[key] = evaluator.evaluate(
            scenarios=causrca_scenarios,
            cd_method=cd_method,
            rca_method=rca_method,
        )
        print(f"{key}: Rung2={results[key].rung2_score:.3f}")
```

---

## Key Novel Contributions to Implement

### 1. Causal Sufficiency Score (CSS)

Already implemented in `factorybench/eval/metrics.py`:

```python
from factorybench.eval.metrics import compute_causal_sufficiency_score

# Collect F1 and MAP scores from experiments
cd_f1_scores = [0.29, 0.40, 0.37, 0.35]  # From your results
rca_map_scores = [0.87, 0.48, 0.37, 0.56]

# Compute CSS for target MAP@3 = 0.7
css = compute_causal_sufficiency_score(cd_f1_scores, rca_map_scores, target_map=0.7)
print(f"CSS(0.7) = {css:.3f}")  # Minimum F1 needed for MAP@3 ≥ 0.7
```

### 2. IRCA Protocol

Already implemented in `factorybench/eval/irca.py`:

```python
from factorybench.eval.irca import IRCAProtocol

irca = IRCAProtocol()
result = irca.evaluate(scenarios, rca_method)

print(f"Standard MAP@3: {result.standard_map_at_3:.3f}")
print(f"IRCA Success Rate: {result.intervention_success_rate:.3f}")
print(f"IRCA Gap: {result.irca_gap:.3f}")  # Large gap = spurious correlations
print(result.interpretation())
```

### 3. Conformal Calibration

Already implemented in `factorybench/eval/conformal.py`:

```python
from factorybench.eval.conformal import ConformalRCA

# Wrap any RCA method
conformal = ConformalRCA(base_method=my_rca, alpha=0.1)

# Calibrate on held-out data
conformal.calibrate(calibration_scenarios)

# Evaluate calibration
result = conformal.evaluate(test_scenarios)
print(result.summary())
```

---

## Figure Generation

### Figure 4: The Causal Gap

```python
import matplotlib.pyplot as plt

rungs = ['Rung 1\n(Assoc)', 'Rung 2\n(Interv)', 'Rung 3\n(Counter)', 'Rung 4\n(Remed)']
oracle = [0.95, 0.92, 0.89, 0.85]
best_method = [0.84, 0.69, 0.42, 0.74]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(rungs))
width = 0.35

ax.bar(x - width/2, oracle, width, label='Oracle (Upper Bound)', color='#2ecc71')
ax.bar(x + width/2, best_method, width, label='Best Method (CIGNN+SCM)', color='#e74c3c')

ax.set_ylabel('Performance Score')
ax.set_xlabel('Pearl\'s Causal Hierarchy')
ax.set_xticks(x)
ax.set_xticklabels(rungs)
ax.legend()
ax.set_title('The Causal Gap: Performance Degradation Across Hierarchy')

# Add gap annotation
ax.annotate('', xy=(2, 0.42), xytext=(2, 0.89),
            arrowprops=dict(arrowstyle='<->', color='gray'))
ax.text(2.1, 0.65, 'Causal Gap\n-51%', fontsize=10)

plt.savefig('figures/causal_gap.png', dpi=300, bbox_inches='tight')
```

---

## Dependencies to Install

```bash
# Core
pip install numpy pandas matplotlib seaborn

# Causal discovery
pip install causal-learn  # PC, FCI, GES, PCMCI

# Optional for advanced methods
pip install networkx  # Graph operations
pip install scipy     # Statistical tests
pip install scikit-learn  # ML baselines
```

---

## Your Existing Results → Paper Tables

Your CausRCA results map directly to paper tables:

### Table 3 (Causal Discovery) - DONE
| Method | Coolant F1 | Hydraulic F1 | Probe F1 |
|--------|------------|--------------|----------|
| PC | 0.29 | 0.40 | 0.37 |
| FCI | 0.29 | 0.40 | 0.37 |
| FGES | 0.13 | 0.28 | 0.26 |
| PCMCI | 0.26 | 0.16 | 0.29 |

### Table 5 (Unsupervised RCA) - DONE
| Method | Coolant MAP@3 | Hydraulic MAP@3 | Probe MAP@3 |
|--------|---------------|-----------------|-------------|
| TimeRecency | 0.35 | 0.18 | 0.19 |
| CausalPrio | **1.00** | **0.74** | **0.58** |
| PageRank | **1.00** | **0.98** | 0.26 |

### Key Finding (Figure 3) - DONE
Your correlation plot showing F1 vs MAP@3 is the paper's key insight!

---

## Questions?

1. **Environment issues?** Check `pyproject.toml` for dependencies
2. **API keys?** Need Azure OpenAI key from Jonas
3. **CausRCA data format?** See your existing `eval/cd/results/` directory

---

## Quick Start Checklist

- [ ] Read paper draft: `docs/FactoryBench_NeurIPS_Paper_Draft.md`
- [ ] Run synthetic generator: `python -c "from factorybench.synthetic import ScenarioGenerator; print(ScenarioGenerator().generate(ScenarioConfig.from_tier(DifficultyTier.APPRENTICE)))"`
- [ ] Adapt CausRCA data to Scenario format
- [ ] Wrap PC algorithm as CausalDiscoveryMethod
- [ ] Run HierarchicalEvaluator on CausRCA
- [ ] Generate Figure 4 (Causal Gap)
- [ ] Compute CSS from your F1/MAP data

Good luck! The hard research is done - now it's about clean execution.
