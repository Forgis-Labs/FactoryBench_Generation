# Archived Code

This folder contains code from the **deprecated causal hierarchy approach** (Pearl's Ladder, Rungs 1-4).

**Why archived:** The paper pivoted from "causal troubleshooting benchmark" to **"Q&A benchmark for machine understanding"**. The Q&A framework is more concrete, leverages LLM infrastructure directly, and enables clearer evaluation.

## What's Here

```
archive/
└── causal_framework/
    ├── FactoryBench_NeurIPS_Paper_Draft_v1_causal.md   # Original paper (54k chars)
    ├── IMPLEMENTATION_GUIDE.md                         # Original implementation plan
    ├── eval/
    │   ├── hierarchical.py      # HierarchicalEvaluator (Pearl's rungs)
    │   ├── metrics.py           # Rung 2-4 metrics (SHD, CSS, F1)
    │   ├── irca.py              # Interventional RCA Protocol
    │   └── conformal.py         # Conformal prediction calibration
    └── synthetic/
        ├── __init__.py
        ├── config.py            # Scenario difficulty tiers
        ├── scenario.py          # SCM dataclasses
        └── generator.py         # Causal graph generator
```

## Potentially Reusable

Some concepts may still be useful:
- **Conformal calibration** (`conformal.py`) - uncertainty quantification for any predictor
- **Scenario generation patterns** - could adapt for Q&A pair generation
- **Metric computation utilities** - F1, precision/recall implementations

## Do Not Use

- `hierarchical.py` - designed for causal discovery methods, not LLM Q&A
- `irca.py` - requires intervention simulation, not applicable to Q&A
- `metrics.py` - Rung-specific metrics (CSS, SHD) not relevant to Q&A levels

---

*Archived: February 2026*
*New direction: See `docs/FactoryBench_NeurIPS_Paper_Draft.md`*
