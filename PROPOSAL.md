# FactoryBench: Q&A Benchmark towards general Machine Understanding

## 1. Project Overview

FactoryBench is a question answering benchmark built on top of multivariate time series data from industrial one armed cobots.  
The goal is to evaluate machine understanding across increasing levels of reasoning about robotic systems.

The dataset is constructed from:

- **Primary source**: FactoryCellData, a dense robotics dataset
- **Secondary source**: Open datasets when suitable
  - Currently: AURSAD
- **Simulation data**: Used for scalability, control, and precise intervention

The benchmark evaluates models on progressively harder reasoning tasks grounded in real robotic behavior.

---

## 2. Levels of Machine Understanding

We define four levels of understanding:

### Level 1: State

Questions about the current state of the robot or environment.

Examples:

- Based on the provided timeseries, describe the kinematic state of the robot at t=2.3s. Which specific parts of the arm are in motion?

  A: The robot is actively moving its lower arm structure while keeping its wrist orientation fixed. Specifically, joints 0, 1, and 2 exhibit non-zero velocity. Joints 3, 4, and 5 are completely stationary (zero velocity).

- Analyze the trajectory from t=1.0s to t=3.0s. What phase of a pick-and-place operation does this represent, and what is the physical evidence in the joint behavior?

  A: This represents the approach phase. The physical evidence is that joints 0, 1, and 2 have active setpoint velocities to translate the tool center point (TCP) through space, while joints 3, 4, and 5 maintain a constant position, indicating the wrist is holding a fixed orientation as it approaches the target.

- The metadata states the robot is moving unloaded. Does the physical data support this? Explain your reasoning.

  A: No, the physical data contradicts the metadata. The effort (current) on joint 1 and joint 2 is elevated by approximately 18% compared to an unloaded baseline for this specific pose. This constant gravitational torque offset indicates the robot is carrying an undeclared load of approximately 2kg.

Focus:

- Direct interpretation of time series + engineering knowledge

---

### Level 2: Intervention

Understanding consequences of an observed intervention.

Structure:

- An event occurs at time T
- Questions are about the state _before_ the event
- Ground truth is derived from what happens _after_ the event

Focus:

- Causal reasoning from intervention
- Event conditioned behavior analysis

Example:

- Q: What will be the output rate if a screw comes unloose now?

  A: Output rate would drop to 50 g/s.

---

### Level 3: Counterfactual

Reasoning about alternative outcomes under controlled variations.

Structure for data generation in FactoryCellData:

- A benchmark time series of a robot performing a movement
- The same movement is repeated 3 to 5 times
- An event is injected at the exact same time of each repetition
- KL divergence is computed between sub series before the event and the benchmark
- The series with minimal KL divergence is selected as ground truth

Focus:

- Controlled counterfactual comparison
- Distribution matching before intervention
- Isolation of event effects

Example:

- Q: What would be the output rate if a screw had come unloose at time t=30ms?

  A: Output rate would be to 30 g/s.

---

### Level 4: Decision

Generally under the format data+prompt -> output steps. In our dataset, we are currently focusing on troubleshooting steps, and maybe optimization eventually.

Structure on data generation:

- A root cause is injected during execution
- The root cause is mapped to a predefined set of valid next steps
- These steps are used as ground truth

---

## 3. Answer Formats

Each question belongs to one of five answer types:

1. Multi select multiple choice
2. Scalar value
3. Tensor
4. Ranking
5. Free form

### Answer Format by Level

| Answer Format | Level 1: State | Level 2: Intervention | Level 3: Counterfactual | Level 4: Decision |
| ------------- | -------------- | --------------------- | ----------------------- | ----------------- |
| Multi select  | ✓              | ✓                     | ✓                       | ✓                 |
| Scalar        | ✓              | ✓                     | ✓                       | -                 |
| Tensor        | ✓              | ✓                     | ✓                       | -                 |
| Ranking       | ✓              | ✓                     | ✓                       | ✓                 |
| Free form     | ✓              | ✓                     | ✓                       | ✓                 |

Aiming for 4-5 question templates on average per cell.

### Evaluation

- Multi select, scalar, tensor, and ranking are easily evaluated, but remain way harder than TF and MC (used in TSAQA)
- Free form is evaluated using a voting mechanism:
  - Multiple state of the art models evaluate whether a candidate answer matches the ground truth
  - Majority voting determines correctness

---

## 4. Question Generation

Questions are generated using:

- Prebuilt templates (level specific)
- Variable placeholders (time, axis, thresholds, intervals, etc.)

We might use LLMs to reformulate the questions rather than raw synthetic format, which I've seen affect negatively finetuning of LLM models in some papers.

---

## 5. Ground Truth Generation

### 5.1 State Questions

Ground truth is generated automatically using:

- Deterministic computation from time series
- Metadata not present in the raw robot stream:
  - Fault labels
  - Event timestamps
  - Task identifiers
  - Robot movement type
  - (Maybe) image data or manual knowledge

While the metadata we have on open datasets is limited, FactoryCellData (which is quite dense) should hopefully close the gaps. Simulations are the most controllable and scalable but data quality isn't as good as real data.

---

### 5.2 Intervention Pipeline

#### Visual Representation

```
Robot Execution Timeline:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          Pre-Event Data              Post-Event Data
├────────────────────────────┤  ├────────────────────┤
0                           T  │                    End
                               ▼
                         Event Injection
                      (e.g., screw loosens)

         ┌─────────────────────┴───────────────────┐
         │                                         │
         ▼                                         ▼
  Question Generation                    Ground Truth Extraction
  "What will happen?"                    "Output drops to 50 g/s"
```

Procedure:

1. Run robot execution
2. Inject a controlled event at time T
3. Record full time series
4. Use:
   - Data before event → question generation
   - Data after event → ground truth

---

### 5.3 Counterfactual Pipeline

#### Visual Representation

```
Benchmark Execution (no event):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            T
                            │

Repetition 1 (event at T):
━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━
                            ▼ Event     KL(pre) = 0.023  ← Selected!

Repetition 2 (event at T):
━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━
                            ▼ Event     KL(pre) = 0.087

Repetition 3 (event at T):
━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━
                            ▼ Event     KL(pre) = 0.045

        Compare pre-event distributions
                    ↓
        Select minimum KL divergence
                    ↓
            Use as ground truth
```

Procedure:

1. Record a benchmark execution of a movement
2. Repeat the same movement 3 to 5 times
3. Inject an event at the exact same timestamp
4. Compute KL divergence between:
   - Pre event sub series of each run
   - Pre event sub series of benchmark
5. Select the run with minimal KL divergence
6. Use that run as ground truth

Rationale:

- Ensures comparable pre event distributions
- Minimizes confounding variability
- Isolates effect of injected event

---

### 5.4 Decision Pipeline

#### Visual Representation

```
Execution with Root Cause:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Normal Operation    │    Fault Manifests
├──────────────────────────────┤├───────────────────┤
0                            Inject                End
                         Root Cause
             (e.g., physical contact with machine)
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Full Trajectory    │
                    │      Recording       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Root Cause Mapping  │
                    │    Database/Rules    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Valid Action Steps  │
                    │   (Ground Truth)     │
                    └──────────────────────┘
                               │
                      • Check compressor
                      • Inspect lines
                      • Verify regulator
                      • Replace sensor
```

Procedure:

1. Inject a known root cause during execution
2. Record full trajectory
3. Map root cause to:
   - Allowed corrective actions
   - Valid next operational steps
4. Use mapped steps as ground truth

This enables:

- Root cause aware reasoning
- Action policy evaluation

---

## 6. Event and Root Cause Tracking

For IR and CF pipelines, it is mandatory to:

- Log event type
- Log exact event timestamp
- Track severity
- Associate events with root causes
- Maintain consistent metadata structure

Without precise timing control, causal evaluation becomes invalid.

---

## 7. Summary

FactoryBench is a structured Q&A benchmark for robotic machine understanding that:

- Combines real datasets, open datasets, and simulation
- Covers four reasoning levels from state to decision
- Supports five answer formats
- Uses deterministic evaluation wherever possible
- Applies model voting for free form evaluation
- Uses controlled intervention and counterfactual pipelines to ensure causal validity

The result is a scalable, controllable, and causally grounded benchmark for evaluating machine reasoning over industrial time series data.
