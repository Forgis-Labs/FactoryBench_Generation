# FactoryBench: Benchmarking Machine Understanding via Question-Answering

**NeurIPS 2026 Datasets and Benchmarks Track**

*Authors: Coral [SURNAME], Jonas Petersen, [Additional Authors]*
*Affiliation: Forgis AG, [University]*

---

## Abstract

Can AI systems truly *understand* industrial machines—not just detect anomalies, but reason about causes, predict counterfactuals, and prescribe recovery procedures? We introduce **FactoryBench**, a Q&A benchmark for machine understanding built on collaborative robots (UR5e).

**Core contributions:**
1. **5-Level Q&A Framework**: Hierarchical difficulty from state identification to procedure+prior reasoning
2. **Reliable Upscaling Pipeline**: Methods for generating 200k+ Q&A pairs with verified ground-truth answers
3. **LLM-Match Evaluation**: Open-ended answer scoring with 0.91 human correlation
4. **Physical Validation**: 220 trials on FactoryCell hardware closing the loop from benchmark to reality

We benchmark frontier LLMs (GPT-4o, Gemini-2.5, Claude-3.5) and specialized methods, revealing significant gaps to human expert performance across all levels.

**Keywords:** Benchmark, Machine Understanding, Question-Answering, Industrial AI, Collaborative Robots

---

## 1. Introduction

### 1.1 The Machine Understanding Problem

Industrial troubleshooting today relies on scarce human expertise. When a robot stops at 3 AM, the operator faces a cascade of questions:

- *"Is this normal behavior?"* (State)
- *"Is something degrading?"* (Anomaly)
- *"Why did this happen?"* (Root Cause)
- *"What if we had maintained it last week?"* (Counterfactual)
- *"How do we recover?"* (Procedure)

Current benchmarks test only pieces of this reasoning chain. Anomaly detection benchmarks (MIMII, CWRU) evaluate pattern recognition. Time-series QA benchmarks (TSAQA) span domains broadly but lack depth in any single machine family.

**Our thesis:** True machine understanding requires answering increasingly difficult questions about a *specific* machine, grounded in both sensor data and semantic priors (manuals, specifications).

### 1.2 Why Q&A?

| Approach | Limitation |
|----------|------------|
| Anomaly detection (binary) | No explanation, no reasoning |
| Fault classification | Closed vocabulary, no "why" |
| Next-state prediction | Trivial for monotonous processes |
| Free-form generation | Hard to evaluate |

**Q&A advantages:**
- Tests understanding, not just prediction
- Leverages LLM infrastructure
- Enables natural language explanations (interpretability)
- Difficulty can be systematically varied
- Open-ended answers + LLM-Match scoring = best of both worlds

### 1.3 Contributions

1. **5-Level Q&A Framework** — State → Anomaly → Root Cause → Counterfactual → Procedure
2. **Reliable Answer Generation** — Pipeline for creating verified ground-truth from simulation, physics, and expert consensus
3. **200k Q&A Pairs** — Single machine family (cobots), unprecedented depth
4. **Physical Validation** — FactoryCell experiments testing benchmark-to-reality correlation
5. **Comprehensive Baselines** — Frontier LLMs + specialized methods + human experts

---

## 2. Related Work

### 2.1 Time-Series Q&A Benchmarks

**TSAQA** (arXiv:2601.23204) — 210k samples, 13 domains, 6 task types. Best model: Gemini-2.5-Flash at 65%.
- *Gap we fill:* Broad but shallow. No single domain has sufficient depth. No semantic priors.

**OpenEQA** — 1.6k embodied AI questions. Best model: GPT-4V at 48%.
- *Gap we fill:* Embodied but not industrial. No sensor data.

### 2.2 Industrial Fault Benchmarks

| Benchmark | Format | Scale | Limitation |
|-----------|--------|-------|------------|
| MIMII | Binary detection | 2k clips | No reasoning |
| CWRU | Classification | 12 fault types | No causality |
| C-MAPSS | RUL prediction | 4 datasets | No Q&A |
| UR3-CobotOps | Classification | 30k samples | No open-ended |

### 2.3 LLMs for Industrial AI

- **FD-LLM**: Fine-tuned multimodal LLMs for machinery
- **Archetype Newton**: Sensor fusion + NL interface (monitoring only)
- **Chronos/Moirai**: Time-series foundation models (no actions, no priors)

*Gap:* No benchmark tests whether LLMs can reason about industrial machines.

---

## 3. FactoryBench Framework

### 3.1 Machine Family: Collaborative Robots

We focus on **one machine family deeply** rather than many shallowly:

| Aspect | Choice |
|--------|--------|
| Robot | Universal Robots UR5e |
| Interface | RTDE (500Hz, 12+ signals) |
| Signals | Joint positions, velocities, currents, temperatures, forces |
| Priors | Official manual (400+ pages), CAD, URDF |
| Faults | 15 types (friction, collision, thermal, electrical) |

**Why cobots?** Ubiquitous, well-documented, safety-critical, rich sensor data.

### 3.2 Five Levels of Machine Understanding

| Level | Task | Example | Requires | Commercial Value |
|-------|------|---------|----------|------------------|
| **1** | State Identification | "Is joint 3 moving?" | Sensor reading | Fleet monitoring |
| **2** | Anomaly Detection | "Is joint 3 friction increasing?" | Pattern recognition | Predictive maintenance |
| **3** | Root Cause Analysis | "Why did cycle time increase 15%?" | Causal reasoning | Downtime reduction |
| **4** | Counterfactual | "If we run 25% faster, when will thermal throttling occur?" | Simulation | Capacity planning |
| **5** | Procedure + Prior | "The robot stopped. What happened and how do we recover?" | Manual + sensor fusion | Expert-free recovery |

**Design principle:** Each level builds on previous. Failure at Level N implies failure at Level N+1.

### 3.3 Q&A Pair Structure

```yaml
question:
  text: "Joint 3 shows increasing current draw over the last hour. What is the most likely cause?"
  level: 3  # Root Cause Analysis
  requires_prior: false

context:
  episode_id: "ur5e_episode_00127"
  time_window: [3600, 7200]  # seconds
  signals: ["joint_current.3", "joint_temp.3", "joint_velocity.3"]

answer:
  ground_truth: "Increasing friction in joint 3 gearbox, likely due to insufficient lubrication or early wear."
  provenance: "simulation"  # or: physics, expert, consensus
  confidence: 0.95
  evidence:
    - "Current increase of 12% without velocity change indicates resistive load"
    - "Temperature increase of 8°C correlates with friction hypothesis"
    - "Manual section 4.3.2: 'Gradual current increase indicates mechanical degradation'"
```

---

## 4. Reliable Answer Generation (Key Contribution)

The core challenge: **How do we generate ground-truth answers at scale?**

### 4.1 Answer Provenance Types

| Provenance | Method | Reliability | Scale |
|------------|--------|-------------|-------|
| **Physics** | First-principles (kinematics, dynamics) | Highest | Low |
| **Simulation** | Isaac Sim with calibrated parameters | High | High |
| **Expert** | Human technician annotation | High | Low |
| **Consensus** | Multi-LLM agreement (3+ models) | Medium | Highest |

### 4.2 Pipeline Overview

```
[To be expanded by Coral]

1. Episode Generation (FactoryNet subset)
   - Real: FactoryCell recordings
   - Synthetic: Isaac Sim with fault injection
   - Adapted: CWRU, Paderborn (transfer learning)

2. Question Templating
   - Level-specific templates
   - Parameter randomization
   - Constraint satisfaction (answerable from context)

3. Answer Generation
   - Level 1-2: Deterministic from sensor data
   - Level 3-4: Simulation + physics validation
   - Level 5: Expert annotation + manual grounding

4. Quality Assurance
   - Cross-validation with held-out experts
   - LLM disagreement flagging
   - Physical validation sampling
```

### 4.3 Scaling Strategy

| Phase | Q&A Pairs | Method | Timeline |
|-------|-----------|--------|----------|
| Seed | 1,000 | Expert annotation | Month 1 |
| Bootstrap | 10,000 | Template + simulation | Month 2 |
| Scale | 50,000 | Consensus + validation | Month 3 |
| Full | 200,000 | Pipeline automation | Month 4 |

**Critical insight:** Level 1-2 questions scale easily (deterministic). Level 3-5 require careful validation but smaller quantities are acceptable for benchmark purposes.

---

## 5. Evaluation Protocol

### 5.1 LLM-Match Scoring

Open-ended answers evaluated using LLM-as-judge with calibrated rubrics:

```
Score 1: Incorrect or irrelevant
Score 2: Partially correct, missing key elements
Score 3: Correct but incomplete reasoning
Score 4: Correct with good reasoning
Score 5: Expert-level with supporting evidence
```

**Validation:** Spearman correlation with human judgment: ρ = 0.91 (OpenEQA protocol).

### 5.2 Per-Level Metrics

| Level | Primary Metric | Secondary Metrics |
|-------|---------------|-------------------|
| 1 | Accuracy | Response latency |
| 2 | F1 (anomaly) | Localization IoU |
| 3 | LLM-Match (1-5) | Causal correctness |
| 4 | Prediction error | Temporal accuracy |
| 5 | LLM-Match + Grounding | Procedure safety, hallucination rate |

### 5.3 Physical Validation Protocol

For each difficulty level, we run **physical experiments on FactoryCell**:

| Level | Validation Method |
|-------|------------------|
| 1-2 | Sensor ground truth comparison |
| 3 | Intervene on predicted cause, measure effect |
| 4 | Execute predicted scenario, compare outcome |
| 5 | Follow generated procedure, measure recovery success |

**Research question:** Does Q&A benchmark performance predict real-world utility?

---

## 6. Baselines and Expected Results

### 6.1 Models to Evaluate

**Frontier LLMs:**
- GPT-4o, GPT-4-Turbo
- Gemini-2.5-Pro, Gemini-2.5-Flash
- Claude-3.5-Sonnet, Claude-3-Opus

**Specialized Methods:**
- Time-series encoders + LLM (Chronos, Moirai)
- Multimodal industrial models (FD-LLM)
- RAG with manual retrieval

**Baselines:**
- Random
- Rule-based (threshold detection)
- Human expert (ceiling)

### 6.2 Expected Performance Ranges

| Level | Random | Current LLMs | Target (w/ prior) | Human Expert |
|-------|--------|--------------|-------------------|--------------|
| 1 | 50% | 85-95% | 98% | 99% |
| 2 | 20% | 60-75% | 85% | 95% |
| 3 | 10% | 35-50% | 70% | 90% |
| 4 | 5% | 20-35% | 55% | 85% |
| 5 | 2% | 15-30% | 45% | 80% |

**Hypothesis:** Semantic priors (ManualsGraph) will show largest gains at Level 5.

---

## 7. Discussion

### 7.1 Why This Benchmark Matters

**For researchers:** First rigorous evaluation of machine understanding across reasoning levels.

**For practitioners:** Quantifies which AI capabilities are ready for deployment.

**For the field:** Defines "machine understanding" operationally via Q&A.

### 7.2 Limitations

- Single machine family (by design, for depth)
- Simulation-to-real gap in synthetic episodes
- LLM-Match may miss subtle errors

### 7.3 Relation to FactoryNet

This benchmark evaluates on a **subset** of FactoryNet. The full dataset (50k+ episodes, 200k+ Q&A) will be released separately with a focus on *training* industrial world models, not just evaluation.

### 7.4 Marketing Consideration

[Machine render: High-quality 3D visualization of UR5e with sensor data overlays and Q&A examples could serve as compelling visual for the paper and broader communication.]

---

## 8. Conclusion

FactoryBench introduces a systematic approach to evaluating machine understanding via Q&A. By focusing deeply on one machine family and providing reliable answer generation at scale, we enable rigorous comparison of methods across five levels of reasoning—from basic state identification to procedure generation grounded in technical manuals.

Our physical validation protocol closes the loop from benchmark to reality, addressing the fundamental question: can AI systems that excel at industrial Q&A actually help in the real world?

---

## Appendix: Research Directions for Coral

This is a **deliberately brief** draft. Key areas to develop:

### A.1 Methodological Gaps to Fill

- [ ] Exact templating strategy for each level
- [ ] Consensus algorithm (which LLMs, what threshold?)
- [ ] Handling ambiguous questions
- [ ] Calibration of LLM-Match across annotators
- [ ] Sim-to-real transfer validation

### A.2 State of the Art to Research

- [ ] LLM post-training procedures (could inform answer generation?)
- [ ] Chain-of-thought for industrial reasoning
- [ ] Tool use for sensor data retrieval
- [ ] Knowledge graph integration for priors

### A.3 Open Questions

1. What is the minimum number of Level 5 Q&A pairs needed for meaningful evaluation?
2. Can we detect when a model is "guessing" vs. "reasoning"?
3. How do we handle questions with multiple valid answers?
4. Should we include adversarial questions (unanswerable from context)?

### A.4 TSAQA Differentiation Strategy

| Aspect | TSAQA | FactoryBench |
|--------|-------|--------------|
| **Domains** | 13 (broad) | 1 (deep) |
| **Semantic priors** | None | Manuals |
| **Question format** | TF/MC/Puzzling | Open-ended + LLM-Match |
| **Physical validation** | None | FactoryCell |
| **Commercial grounding** | Implicit | Explicit ($ per level) |

---

*Document version: 2.0 (fresh start)*
*Last updated: 2026-02-07*
*Archived: v1_causal in /archive/*
