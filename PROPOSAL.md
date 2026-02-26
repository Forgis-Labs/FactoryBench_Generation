# FactoryBench: Q&A Pipeline for Machine Understanding

## 1. Project Overview

FactoryBench is a question answering benchmark built on top of multivariate time series data from industrial one armed cobots.  
The goal is to evaluate machine understanding across increasing levels of reasoning about robotic systems.

The dataset is constructed from:

- **Primary source**: FactoryCell, a dense robotics dataset
- **Secondary source**: Open datasets when suitable
  - Currently: AURSAD
- **Simulation data**: Used for scalability, control, and precise intervention

The benchmark evaluates models on progressively harder reasoning tasks grounded in real robotic behavior.

---

## 2. Levels of Machine Understanding

We define four levels of understanding:

### Level 1: State Understanding

Questions about the current state of the robot or environment.

Examples:

- What is the velocity of end at time T?
- Is the behaviour of the machine abnormal?
- What features of the machine are behaving abnormally?

Focus:

- Direct interpretation of time series + engineering knowledge

---

### Level 2: Intervention Reasoning (IR)

Understanding consequences of an observed intervention.

Structure:

- An event occurs at time T
- Questions are about the state _before_ the event
- Ground truth is derived from what happens _after_ the event

Focus:

- Causal reasoning from intervention
- Event conditioned behavior analysis

Examples:

- What will be the output rate if a screw comes unloose now?

---

### Level 3: Counterfactual Reasoning (CF)

Reasoning about alternative outcomes under controlled variations.

Structure for data generation in FactoryCell:

- A benchmark time series of a robot performing a movement
- The same movement is repeated 3 to 5 times
- An event is injected at the exact same time of each repetition
- KL divergence is computed between sub series before the event and the benchmark
- The series with minimal KL divergence is selected as ground truth

Focus:

- Controlled counterfactual comparison
- Distribution matching before intervention
- Isolation of event effects

---

### Level 4: Decision Making (DM)

Generally under the format data+prompt -> output steps. In our dataset, we are currently focusing on troubleshooting steps, and maybe optimization eventually.

Structure on data generation:

- A root cause is injected during execution
- The root cause is mapped to a predefined set of valid next steps
- These steps are used as ground truth

Focus:

- Diagnostic reasoning
- Action selection
- Planning under fault conditions

---

## 3. Answer Formats

Each question belongs to one of five answer types:

1. Multi select multiple choice
2. Scalar value
3. Tensor
4. Ranking
5. Free form

Evaluation:

- Multi select, scalar, tensor, and ranking are easily evaluated, but remain way harder than TF and MC (used in TSAQA)
- Free form is evaluated using a voting mechanism:
  - Multiple state of the art models evaluate whether a candidate answer matches the ground truth
  - Majority voting determines correctness

---

## 4. Question Generation

Questions are generated using:

- Prebuilt templates
- Level specific question formats
- Variable placeholders (time, axis, thresholds, intervals, etc.)

This ensures:

- Structural consistency
- Controlled difficulty
- Automatic scalability

We might use LLMs to reformulate the questions rather than raw synthetic format, which I've seen affect negatively finetuning of LLM models in some paper.

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

While the metadata we have on open datasets is limited, FactoryCell (which is quite dense) should hopefully close the gaps. Simulations are the most controllable and scalable but data quality isn't as good as real data.

---

### 5.2 Intervention Reasoning Pipeline

Procedure:

1. Run robot execution
2. Inject a controlled event at time T
3. Record full time series
4. Use:
   - Data before event → question generation
   - Data after event → ground truth

Requirement:

- Precise tracking of event timestamps
- Logging of event type and severity

---

### 5.3 Counterfactual Pipeline

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

Requirement:

- Event timestamp tracking
- Sub series extraction
- Distribution comparison

---

### 5.4 Decision Making Pipeline

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
- Covers four reasoning levels from state to decision making
- Supports five answer formats
- Uses deterministic evaluation wherever possible
- Applies model voting for free form evaluation
- Uses controlled intervention and counterfactual pipelines to ensure causal validity

The result is a scalable, controllable, and causally grounded benchmark for evaluating machine reasoning over industrial time series data.
