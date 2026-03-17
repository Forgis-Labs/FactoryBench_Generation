# FactoryBench

**Q&A benchmark for machine understanding** — evaluating whether AI can reason about industrial machines, not just detect anomalies.

**Target:** NeurIPS 2026 Datasets and Benchmarks Track

---

## The Idea

When a robot stops at 3 AM, operators face a cascade of questions:
- *"What is happening right now?"* (State)
- *"What if I do this now?"* (Intervention)
- *"What if I had done that in the past?"* (Counterfactual)
- *"How do we recover?"* (Decision making)

**FactoryBench tests all four levels** on collaborative robots (UR5e), going deeper on one machine family than existing benchmarks go across many.

---

> 📝 **Paper Draft:** Our NeurIPS paper draft for FactoryBench is actively being written in [`docs/FactoryBench_NeurIPS_Paper_Draft.md`](docs/FactoryBench_NeurIPS_Paper_Draft.md).

---
## 4-Level Q&A Framework: Levels of Understanding

To systematically evaluate machine understanding, we organize question-answering tasks according to a four-tier hierarchy, each probing distinct reasoning capabilities:

* **Level 1: State**. This tier assesses the agent’s ability to interpret the current state of the machine, including detection of anomalies, identification of operational modes, and recognition of sensor patterns. Questions at this level require accurate extraction and interpretation of time series features.
* **Level 2: Intervention**. At this level, the agent must reason about the consequences of interventions or events occurring at the present timestep that might perturb the distribution of machine states. Tasks include predicting the immediate impact of control actions, diagnosing faults as they arise, and understanding causal relationships in real time.
* **Level 3: Counterfactual**. This tier evaluates the agent’s ability to reason about hypothetical scenarios, such as the effect of an event or intervention at a previous timestep. Questions require the agent to simulate alternative histories and assess how outcomes would differ under counterfactual conditions, while still considering the history they know.
* **Level 4: Decision Making**. The highest tier encompasses complex decision-making tasks, where the agent must generate a sequence of actions or recommendations based on the time series and a prompt. This includes troubleshooting, optimization, and planning, requiring integration of state interpretation, causal reasoning, and goal-directed synthesis.

By structuring Q&A tasks along these four levels, FactoryBench enables rigorous and granular assessment of machine understanding, from basic state recognition to advanced decision support.

| Level | Task | Example Question | Ground Truth Source | Pearl Rung |
|-------|------|------------------|---------------------|------------|
| **1** | State | "What's the current of joint 3 now?" | Sensor data (time-series window) | Rung 1 |
| **2** | Intervention | "If force in joint 3 increases *now* to X, what happens?" | Simulation (present state injection) | Rung 2 |
| **3** | Counterfactual | "If force had increased to X *at t=20ms*, what would have happened?" | Simulation (past state replay) | Rung 3 |
| **4** | Decision Making | "Robot stopped with error C203A. What to do?" | Manual + sensor fusion | Composite |

### Answer Format by Level

| Answer Format | Level 1: State | Level 2: Intervention | Level 3: Counterfactual | Level 4: Decision |
|---------------|----------------|-----------------------|-------------------------|-------------------|
| Multi select  | ✓              | ✓                     | ✓                       | ✓                 |
| Scalar        | ✓              | ✓                     | ✓                       | -                 |
| Tensor        | ✓              | ✓                     | ✓                       | -                 |
| Ranking       | ✓              | ✓                     | ✓                       | ✓                 |
| Free form     | ✓              | ✓                     | ✓                       | ✓                 |

**Key insight:** Each level builds on previous. Failure at Level N implies failure at Level N+1.

---

## 🚀 Overview

Modern industrial systems generate large volumes of multivariate time-series data from sensors, actuators, and controllers. Traditional time-series models excel at narrow tasks but lack explicit reasoning, while general LLMs show strong reasoning but struggle with dense numerical time series. 

FactoryBench addresses this by offering:
- **Scalable Framework:** 80 structured question templates spanning diverse reasoning scenarios.
- **FactoryWave Dataset:** A dense multivariate dataset from one-armed robotic systems under varied operational conditions and systematically injected anomalies.
- **SCE Causal Schema:** A unified timeline schema explicitly mapping **S**etpoint, **C**ontext, and **E**ffort/Feedback.

## 🧠 Levels of Understanding

FactoryBench targets a four-tier hierarchy to systematically evaluate reasoning skills:
1. **Level 1: State** - Interpret the current state, detect anomalies, and recognize sensor patterns.
2. **Level 2: Intervention** - Reason about immediate consequences of interventions or events occurring at the present timestep.
3. **Level 3: Counterfactual** - Simulate alternative past timelines, assessing how outcomes would differ under hypothetical conditions.
4. **Level 4: Decision Making** - Generate complex troubleshooting steps, optimization plans, and recovery procedures based on sensor fusion.

## 📂 Repository Structure

```text
FactoryBench/
├── data/                                      # Raw and processed datasets
├── datasets/                                  # Local test fixtures
├── docs/                                      # Documentation and draft
│   └── FactoryBench_NeurIPS_Paper_Draft.md    # Current paper (start here)
├── factorybench/                              # Python package
│   ├── adapters/                              # LLM adapters (Azure OpenAI)
│   ├── api/                                   # FastAPI backend
│   ├── data/                                  # Data loaders
│   ├── eval/                                  # Evaluation runner
│   ├── metrics/                               # Scoring (to extend for Q&A)
│   └── viz/                                   # Charts
├── figures/                                   # Diagrams and illustration assets
├── frontend/                                  # Remix web UI
├── generated_data/                            # Outputs from simulators
├── qa_generation/                             # Scripts and templates for Q&A
├── runs/                                      # Benchmark results
├── src/                                       # Source code for scripts/experiments
└── archive/                                   # Deprecated causal framework code
```

## 🛠️ Quick Start

### 1. Setup

```bash
uv venv && uv pip install -e .
cp .env.example .env  # Add Azure OpenAI keys
```

### 2. Run Existing Stage 1 (Telemetry Literacy)
Start backend ( [http://localhost:5173](http://localhost:5173/) )
```bash
# Backend
uvicorn factorybench.api.app:app --reload --port 5173
```
Start frontend ( [http://localhost:3000](http://localhost:3000/) )
```bash
# Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

### 3. Generate Q&A Pairs

> ⚠️ TO BE COMPLETED
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## 📄 License

*(License information to be added)*

## 📚 Citation

*(Once published, add context here on how to cite the FactoryBench paper and dataset)*
