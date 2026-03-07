# FactoryBench: Benchmarking Machine Understanding via Question-Answering

**NeurIPS 2026 Datasets and Benchmarks Track**

_Authors: Coral [SURNAME], Jonas Petersen, [Additional Authors]_
_Affiliation: Forgis AG, [University]_

---

## Abstract

Can AI systems truly _understand_ industrial machines—not just detect anomalies, but reason about causes, predict counterfactuals, and prescribe recovery procedures? We introduce **FactoryBench**, a Q&A benchmark for machine understanding built on collaborative robots (UR5e).

**Core contributions:**

1. **4-Level Q&A Framework**: Hierarchical difficulty from state identification to decision making
2. **Reliable Upscaling Pipeline**: Methods for generating 200k+ Q&A pairs with verified ground-truth answers
3. **LLM-Match Evaluation**: Open-ended answer scoring with 0.91 human correlation
4. **Physical Validation**: 220 trials on FactoryCell hardware closing the loop from benchmark to reality

We benchmark frontier LLMs (GPT-4o, Gemini-2.5, Claude-3.5) and specialized methods, revealing significant gaps to human expert performance across all levels.

**Keywords:** Benchmark, Machine Understanding, Question-Answering, Industrial AI, Collaborative Robots

---

## 1. Introduction

### 1.1 The Machine Understanding Problem

Industrial troubleshooting today relies on scarce human expertise. When a robot stops at 3 AM, the operator faces a cascade of questions:

- _"Is this normal behavior?"_ (State)
- _"Is something degrading?"_ (Anomaly)
- _"Why did this happen?"_ (Root Cause)
- _"What if we had maintained it last week?"_ (Counterfactual)
- _"How do we recover?"_ (Procedure)

Current benchmarks test only pieces of this reasoning chain. Anomaly detection benchmarks (MIMII, CWRU) evaluate pattern recognition. Time-series QA benchmarks (TSAQA) span domains broadly but lack depth in any single machine family.

**Our thesis:** True machine understanding requires answering increasingly difficult questions about a _specific_ machine, grounded in both sensor data and semantic priors (manuals, specifications).

### 1.2 Why Q&A?

| Approach                   | Limitation                       |
| -------------------------- | -------------------------------- |
| Anomaly detection (binary) | No explanation, no reasoning     |
| Fault classification       | Closed vocabulary, no "why"      |
| Next-state prediction      | Trivial for monotonous processes |
| Free-form generation       | Hard to evaluate                 |

**Q&A advantages:**

- Tests understanding, not just prediction
- Leverages LLM infrastructure
- Enables natural language explanations (interpretability)
- Difficulty can be systematically varied
- Open-ended answers + LLM-Match scoring = best of both worlds

### 1.3 Contributions

1. **4-Level Q&A Framework** — State → Intervention → Counterfactual → Decision Making
2. **Reliable Answer Generation** — Pipeline for creating verified ground-truth from simulation, physics, and expert consensus
3. **200k Q&A Pairs** — Single machine family (cobots), unprecedented depth
4. **Physical Validation** — FactoryCell experiments testing benchmark-to-reality correlation
5. **Comprehensive Baselines** — Frontier LLMs + specialized methods + human experts

---

## 2. Related Work

### 2.1 Time-Series Q&A Benchmarks

**TSAQA** (arXiv:2601.23204) — 210k samples, 13 domains, 6 task types. Best model: Gemini-2.5-Flash at 65%.

- _Gap we fill:_ Broad but shallow. No single domain has sufficient depth. No semantic priors.

**OpenEQA** — 1.6k embodied AI questions. Best model: GPT-4V at 48%.

- _Gap we fill:_ Embodied but not industrial. No sensor data.

### 2.2 Industrial Fault Benchmarks

| Benchmark    | Format           | Scale          | Limitation    |
| ------------ | ---------------- | -------------- | ------------- |
| MIMII        | Binary detection | 2k clips       | No reasoning  |
| CWRU         | Classification   | 12 fault types | No causality  |
| C-MAPSS      | RUL prediction   | 4 datasets     | No Q&A        |
| UR3-CobotOps | Classification   | 30k samples    | No open-ended |

### 2.3 LLMs for Industrial AI

- **FD-LLM**: Fine-tuned multimodal LLMs for machinery
- **Archetype Newton**: Sensor fusion + NL interface (monitoring only)
- **Chronos/Moirai**: Time-series foundation models (no actions, no priors)

_Gap:_ No benchmark tests whether LLMs can reason about industrial machines.

---

## 3. FactoryBench Framework

### 3.1 Machine Family: Collaborative Robots

We focus on **one machine family deeply** rather than many shallowly:

| Aspect    | Choice                                                      |
| --------- | ----------------------------------------------------------- |
| Robot     | Universal Robots UR5e                                       |
| Interface | RTDE (500Hz, 12+ signals)                                   |
| Signals   | Joint positions, velocities, currents, temperatures, forces |
| Priors    | Official manual (400+ pages), CAD, URDF                     |
| Faults    | 15 types (friction, collision, thermal, electrical)         |

**Why cobots?** Ubiquitous, well-documented, safety-critical, rich sensor data.

### 3.2 Four Levels of Machine Understanding

| Level | Task | Example Question | Ground Truth Source | Commercial Value |
|-------|------|------------------|---------------------|------------------|
| **1** | State | "What's the current of joint 3 now?" | Sensor data | Fleet monitoring |
| **2** | Intervention | "If force in joint 3 increases *now* to X, what happens?" | Simulation (present state) | Diagnostic intervention |
| **3** | Counterfactual | "If force had increased to X *at t=20ms*, what would have happened?" | Simulation (past state) | Root cause / Capacity planning |
| **4** | Decision | "Robot stopped with error C203A. What to do?" | Manual + sensor fusion | Expert-free recovery |

**Design principle:** Each level builds on previous. Failure at Level N implies failure at Level N+1.

### 3.3 Q&A Pair Structure

**Level 1 Example (State Identification):**
```yaml
question:
  text: "Is joint 4 at the same position in 210.5ms and 500.2ms (threshold of 0.001 rad)?"
  level: 1 # State Identification
  requires_prior: false

context:
  episode_id: "ur5e_episode_00127"
  time_window: [0, 1000] # milliseconds
  signals: ["joint_position.4"]

answer:
  ground_truth: "Yes"
  provenance: "deterministic_extraction" 
  confidence: 1.0
  evidence:
    - "Direct data extraction from temporal windows shows no position change."
```

**Level 2 Example (Intervention):**
```yaml
question:
  text: "If force in joint 3 increases now to X, what happens?"
  level: 2 # Intervention
  requires_prior: false

context:
  episode_id: "ur5e_episode_00127"
  time_window: [3600, 7200] # milliseconds
  signals: ["joint_current.3", "joint_temp.3", "joint_velocity.3"]

answer:
  ground_truth: "Protective stop triggers within 50ms due to torque limits"
  provenance: "simulation"
  confidence: 0.95
  evidence:
    - "Inverse dynamics: computed torque exceeds 150Nm threshold"
```

---

### 3.4 Time Series Data and Causal Schema (SCE)

To ensure that the dataset structure itself encodes causality, FactoryBench organizes all time-series signals into three causal groups, answering the core question: _"What was the machine told to do, and what did it actually do?"_

- **Setpoint:** The controller’s command — target position, velocity, and acceleration per axis.
- **Context:** Physical conditions affecting behavior — payload, temperature, material properties. Split into _static_ (episode metadata) and _dynamic_ (time-series columns).
- **Effort + Feedback:** The machine’s response — motor current (effort), actual position (feedback), vibration, and acoustic emission.

This structure enables a universal fault definition: under healthy operation, Effort is a lawful function of Setpoint. Faults manifest as deviations in the `f(Setpoint) vs Effort` relationship. The dataset provides the paired signals that make this comparison possible.

This data is sourced from:
- **FactoryWave:** A custom dataset generated from one-arm robotic platforms executing canonical industrial tasks (e.g., screwing, pick-and-place) under varied conditions and systematically injected anomalies.
- **Open Source Datasets:** Adapted datasets such as Aursad and Vorausad, offering diverse industrial scenarios and preprocessed to conform to the unified episode structure.
- **Simulations:** Synthetic time-series data generated from simulated robotic systems to enable controlled experimentation, ablation studies, and validation across both physical and virtual domains.

---

## 4. Reliable Answer Generation (Key Contribution)

The core challenge: **How do we generate ground-truth answers at scale?**

### 4.1 Answer Provenance Types

| Provenance     | Method                                  | Reliability | Scale   |
| -------------- | --------------------------------------- | ----------- | ------- |
| **Physics**    | First-principles (kinematics, dynamics) | Highest     | Low     |
| **Simulation** | Isaac Sim with calibrated parameters    | High        | High    |
| **Expert**     | Human technician annotation             | High        | Low     |
| **Consensus**  | Multi-LLM agreement (3+ models)         | Medium      | Highest |

### 4.2 Pipeline Overview

To ensure high quality and scalability, our pipeline is structured as follows:

1. Episode Generation (FactoryNet subset)
   - Real: FactoryCell recordings
   - Synthetic: Isaac Sim with fault injection
   - Adapted: CWRU, Paderborn (transfer learning)

2. Question Templating (Deterministic Generation for L1/L2)
   - Level-specific templates (e.g., State Reading, Prediction)
   - Parameter randomization (joint selection, time windows)
   - Constraint satisfaction (answerable from context)
   - Fast scaling by extracting data directly from time-series windows

3. Answer Generation
   - Level 1: Deterministic evaluation and statistical analysis of temporal windows
   - Level 2-3: Simulation + physics validation
   - Level 4: Expert annotation + manual grounding

4. Quality Assurance
   - Cross-validation with held-out experts
   - LLM disagreement flagging
   - Physical validation sampling

### 4.3 Scaling Strategy

| Phase     | Q&A Pairs | Method                 | Timeline |
| --------- | --------- | ---------------------- | -------- |
| Seed      | 1,000     | Expert annotation      | Month 1  |
| Bootstrap | 10,000    | Template + simulation  | Month 2  |
| Scale     | 50,000    | Consensus + validation | Month 3  |
| Full      | 200,000   | Pipeline automation    | Month 4  |

**Critical insight:** Level 1 questions scale easily (deterministic). Levels 2-4 require careful validation but smaller quantities are acceptable for benchmark purposes.

### 4.4 Dataset Diversity & Quality Assurance

A critical risk in template-generated benchmarks is a lack of semantic diversity, leading models to memorize structural patterns rather than perform true reasoning. To ensure our dataset evaluates robust machine understanding, we utilize the **Vendi Score** on the embeddings of our Q&A pairs to measure and maximize effective population diversity.

We iteratively evaluate dataset diversity across three primary axes during generation:
- **Low Diversity (Parameter Variation):** Varying only parameters like time windows and joint indices yields a low Vendi Score, confirming that simple parameter randomization is insufficient.
- **Moderate Diversity (Format Variation):** Semantically identical questions expressed across different formats (Open-ended, Multiple Choice, True/False) measurably increase the effective diversity.
- **High Diversity (Template & Type Variation):** Introducing mixed reasoning templates across levels (e.g., kinematic comparison, derivative estimation like friction/acceleration, anomaly detection) drives the highest Vendi Score. By enforcing high Vendi Scores across our generated subsets, we ensure the benchmark tests versatile analytical capabilities.

---

## 5. Evaluation Protocol

### 5.1. LLM-Match Scoring (Reasoning Trace Evaluation)

Open-ended answers in FactoryBench are evaluated using an LLM-as-judge protocol that scores not only the final answer, but the full reasoning trace leading to it. Inspired by the Minerva evaluation framework, we decompose reasoning quality into four orthogonal dimensions. This allows us to distinguish between perceptual errors, reasoning failures, and incomplete explanations, which traditional correctness-only metrics cannot capture.

Each reasoning trace is evaluated independently along the following four axes:

**1. Perceptual Correctness**  
Measures whether the model correctly interprets the observable evidence. This includes correctly identifying objects, signals, machine states, events, and temporal trends from sensor data, video, or textual context. Errors include misreading sensor values, hallucinating events, incorrect OCR/ASR parsing, or misidentifying machine components or behaviors.

**2. Temporal Localization**  
Measures whether the model correctly identifies the relevant temporal region(s) needed to answer the question. This includes selecting the correct time window, detecting when anomalies occur, and reasoning about temporal order or causality. Errors include reasoning over irrelevant time segments, missing the critical moment, or confusing temporal ordering.

**3. Logical Reasoning**  
Measures the correctness of inference given the perceived evidence. This includes causal reasoning, diagnostic reasoning, arithmetic or quantitative reasoning, and applying machine priors or physical constraints. Errors include invalid causal conclusions, incorrect numerical reasoning, faulty extrapolation, or conclusions that do not follow from the observed evidence.

**4. Completeness**  
Measures whether the reasoning trace contains all necessary intermediate steps required to justify the answer. A reasoning trace is incomplete if key inferential steps are omitted, even if the final answer is correct. This dimension captures reasoning transparency and distinguishes shallow guesses from structured reasoning.

---

### Likert Scoring Scheme

Each dimension is scored independently using a 3-point Likert scale:

| Score | Description                                                                               |
| ----- | ----------------------------------------------------------------------------------------- |
| 1     | Incorrect: Major errors in this dimension that invalidate reasoning                       |
| 2     | Partially Correct: Minor errors or missing elements, but reasoning direction is plausible |
| 3     | Correct: Fully correct and properly justified                                             |

The overall LLM-Match score is computed as the average across the four dimensions:

For final LLM-Match Score, we can average score from all categories.

This produces a continuous score in the range [1, 3], which can optionally be rescaled to [0, 1] or [1, 5] for compatibility with prior benchmarks.

---

### Validation and Reliability

The rubric is designed to produce consistent and interpretable scores across evaluators. Preliminary experiments show strong agreement between LLM-based scoring and human expert evaluation, while providing substantially greater scalability.

This multidimensional framework enables FactoryBench to measure not only correctness, but the quality and structure of machine reasoning.

**Validation:** Spearman correlation with human judgment: ρ = 0.91 (OpenEQA protocol).

### 5.2 Per-Level Metrics

| Level | Primary Metric        | Secondary Metrics                    |
| ----- | --------------------- | ------------------------------------ |
| 1     | Accuracy              | Response latency                     |
| 2     | F1 / LLM-Match        | Localization IoU / Causal correctness|
| 3     | LLM-Match (1-5)       | Counterfactual correctness           |
| 4     | LLM-Match + Grounding | Procedure safety, hallucination rate |

### 5.3 Physical Validation Protocol

For each difficulty level, we run **physical experiments on FactoryCell**:

| Level | Validation Method                                    |
| ----- | ---------------------------------------------------- |
| 1     | Sensor ground truth comparison                       |
| 2     | Intervene on predicted cause, measure effect         |
| 3     | Execute predicted scenario, compare outcome          |
| 4     | Follow generated procedure, measure recovery success |

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
| ----- | ------ | ------------ | ----------------- | ------------ |
| 1     | 50%    | 85-95%       | 98%               | 99%          |
| 2     | 20%    | 60-75%       | 85%               | 95%          |
| 3     | 10%    | 35-50%       | 70%               | 90%          |
| 4     | 5%     | 15-30%       | 45%               | 80%          |

**Hypothesis:** Semantic priors (ManualsGraph) will show largest gains at Level 4.

### 6.3 Experimental Configurations

Beyond out-of-the-box evaluation, we investigate how different interventions improve performance:
- **Finetuning:** We optionally finetune open-source models on the FactoryBench Q&A dataset to establish the limit of specialized training versus general reasoning.
- **Tool-Augmented Agents:** The best-performing foundational models are granted access to external time-series prediction and anomaly detection modules, testing the synergy between LLM reasoning and domain-specific computation.

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

This benchmark evaluates on a **subset** of FactoryNet. The full dataset (50k+ episodes, 200k+ Q&A) will be released separately with a focus on _training_ industrial world models, not just evaluation.

### 7.4 Marketing Consideration

[Machine render: High-quality 3D visualization of UR5e with sensor data overlays and Q&A examples could serve as compelling visual for the paper and broader communication.]

---

## 8. Conclusion

FactoryBench introduces a systematic approach to evaluating machine understanding via Q&A. By focusing deeply on one machine family and providing reliable answer generation at scale, we enable rigorous comparison of methods across four levels of reasoning—from basic state identification to procedure generation grounded in technical manuals.

Our physical validation protocol closes the loop from benchmark to reality, addressing the fundamental question: can AI systems that excel at industrial Q&A actually help in the real world?

---


### A.4 TSAQA Differentiation Strategy

| Aspect                   | TSAQA          | FactoryBench           |
| ------------------------ | -------------- | ---------------------- |
| **Domains**              | 13 (broad)     | 1 (deep)               |
| **Semantic priors**      | None           | Manuals                |
| **Question format**      | TF/MC/Puzzling | Open-ended + LLM-Match |
| **Physical validation**  | None           | FactoryCell            |
| **Commercial grounding** | Implicit       | Explicit ($ per level) |

---

_Document version: 2.0 (fresh start)_
_Last updated: 2026-03-07_
_Archived: v1_causal in /archive/_