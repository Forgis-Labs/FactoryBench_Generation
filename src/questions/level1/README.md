# Level 1 Question Generator

This module generates **Level 1 (State Identification)** questions for FactoryBench—deterministic Q&A pairs grounded in normalized sensor data from collaborative robots.

## Overview

Level 1 questions test whether an AI system can identify basic operational state from sensor telemetry:

- Joint position changes
- Friction degradation
- End-effector acceleration
- External force detection
- Joint jerk (third derivative)
- Torque magnitude

All answers are **computed directly from sensor data** with no ambiguity—suitable for large-scale benchmark generation.

---

## Files

- **`level1.py`** — Core generator module with 6 question types and answer computation functions
- **`cli.py`** — Command-line interface for batch question generation
- **`phrases_level1.json`** — Question templates with format placeholders
- **`README.md`** — This file

---

## The 6 Question Types

| #   | Question                                                 | Input                  | Answer Type             | Data Source                                      | Threshold     |
| --- | -------------------------------------------------------- | ---------------------- | ----------------------- | ------------------------------------------------ | ------------- |
| Q1  | "Is joint {axis} at same position between t1 and t2?"    | Position + time window | Yes/No                  | `feedback_pos_{axis}`                            | `eps_1` (rad) |
| Q2  | "Has joint {axis} friction increased between t1 and t2?" | Current + speed + time | Yes/No                  | `effort_current_{axis}`, `feedback_speed_{axis}` | `eps_2` (%)   |
| Q3  | "What is end-effector acceleration at t1?"               | Vibration sensors      | Vector [x, y, z] (m/s²) | `vibration_0/1/2`                                | None          |
| Q4  | "Is external force detected at t?"                       | Force/torque sensor    | Yes/No                  | `true_force_*` or `est_contact_force_*`          | `eps_3` (N)   |
| Q5  | "What is jerk of joint {axis} at t?"                     | Joint velocity         | Scalar (rad/s³)         | `feedback_speed_{axis}`                          | None          |
| Q6  | "What is torque magnitude about axis {axis_label} at t?" | Torque sensor          | Scalar (Nm)             | `true_torque_*` or `est_contact_torque_*`        | None          |

### Question Details

#### Q1: Position Check

- **Spec:** Interpolates joint position at t1 and t2, computes displacement Δq = |q(t2) - q(t1)|
- **Answer:** "Yes" if Δq > ε₁, else "No"
- **Reasoning:** Defines t1/t2, shows interpolation mode (exact vs. interpolated), displays delta_q value and comparison
- **Error handling:** Returns "Unknown" if no samples bracket either time point

#### Q2: Friction Increase

- **Spec:** Builds symmetric windows W1 = [t1 - Δ1, t1 + Δ1] and W2 = [t2 - Δ1, t2 + Δ1] (clipped to data range)
- **Friction proxy:** f = median(|I| / |v|) for each window
- **Ratio:** r = f2 / f1, threshold = 1 + ε2 / 100
- **Answer:** "Yes" if r > threshold, else "No"
- **Reasoning:** Shows raw and clipped windows, intermediate f1/f2 values, ratio computation
- **Note:** Uses ALL samples in windows (no steady-motion filtering)

#### Q3: End-Effector Acceleration

- **Spec:** Interpolates vibration_0/1/2 (in g) at t1, converts to m/s² (multiply by 9.81)
- **Answer:** Vector [ax, ay, az] in m/s²
- **Reasoning:** Shows interpolation mode, acceleration values in each axis, magnitude
- **Error handling:** Returns "Unknown" if vibration samples not bracketing t1

#### Q4: External Force Detection

- **Spec:** Gets force vector from wrench sensor (tries external sensor first, falls back to controller estimate)
- **Magnitude:** F = √(Fx² + Fy² + Fz²)
- **Answer:** "Yes" if F ≥ ε3, else "No"
- **Reasoning:** Shows force components, magnitude, threshold comparison
- **Error handling:** Returns "Unknown" if wrench components missing

#### Q5: Joint Jerk

- **Spec:** Finds closest timestamp to t with ±2 samples on each side
- **Method:** Computes acceleration at two points via central differences, then differentiates acceleration
- **Formula:**
  - a₋ = (vk - vk-2) / (tk - tk-2)
  - a₊ = (vk+2 - vk) / (tk+2 - tk)
  - j = (a₊ - a₋) / (tk+1 - tk-1)
- **Answer:** Scalar jerk in rad/s³
- **Reasoning:** Includes note: "Any common derivative estimation is acceptable; here I use central differences."
- **Error handling:** Returns "Unknown" if fewer than 5 timestamped samples or insufficient bracketing

#### Q6: Torque Magnitude

- **Spec:** Axis mapping: x→3, y→4, z→5 (wrench indices)
- **Answer:** Absolute value of torque component in Nm
- **Reasoning:** Shows torque source (external or controller estimate), component value, magnitude
- **Error handling:** Returns "Unknown" if axis_label invalid or torque components missing

---

## Usage

### Quick Start (5 Questions)

```bash
python -m src.questions.cli level1 \
  --input data/normalized_episodes/dummy/ABB.json \
  --output data/questions/dummy/sample_questions.json \
  --n 5
```

### Full Configuration (All Thresholds)

```bash
python -m src.questions.cli level1 \
  --input data/normalized_episodes/dummy/ABB.json \
  --output data/questions/dummy/questions_configured.json \
  --n 100 \
  --eps-1 0.05 \
  --eps-2 15 \
  --delta-1 500 \
  --eps-3 1.0 \
  --seed 42
```

### CLI Arguments

```
positional:
  (none - hardcoded to 'level1')

options:
  --input PATH                Input normalized episode JSON file (required)
  --output PATH               Output questions JSON file (required)
  --n INT                     Number of questions to generate (default: 100)
  --min-dt-ms INT             Minimum time window in milliseconds (default: 100)
  --max-dt-ms INT             Maximum time window in milliseconds (default: 2000)
  --eps-q FLOAT               Legacy threshold in radians (default: 1e-3)
  --eps-1 FLOAT               Q1 threshold: position displacement (rad) (default: eps-q)
  --eps-2 FLOAT               Q2 threshold: friction increase (percent) (default: eps-q)
  --delta-1 INT               Q2 symmetric half-window (ms) (default: max-dt-ms // 2)
  --eps-3 FLOAT               Q4 threshold: force magnitude (N) (default: eps-q)
  --seed INT                  Random seed for reproducibility (optional)
  -v, --verbose               Enable verbose logging
```

### Python API

```python
from pathlib import Path
from src.questions.level1 import generate_level1_questions

questions = generate_level1_questions(
    episode_json=Path("data/normalized_episodes/dummy/ABB.json"),
    out_json=Path("data/questions/dummy/my_questions.json"),
    n_questions=50,
    eps_1=0.05,       # Position threshold (rad)
    eps_2=15.0,       # Friction threshold (%)
    delta_1=500,      # Q2 window half-width (ms)
    eps_3=1.0,        # Force threshold (N)
    seed=42
)
print(f"Generated {len(questions)} questions")
```

---

## Input Format (Normalized Episodes)

Expected JSON structure: **List of dicts with timestamp and signal keys**

```json
[
  {
    "timestamp_ms": 0.0,
    "feedback_pos_0": 0.01,
    "feedback_pos_1": 0.02,
    "..._pos_5": 0.06,
    "feedback_speed_0": 0.001,
    "..._speed_5": 0.005,
    "effort_current_0": 1.2,
    "..._current_5": 1.5,
    "vibration_0": 0.3,
    "vibration_1": 0.25,
    "vibration_2": 0.2,
    "true_force_0": 1.0,
    "true_force_1": 2.0,
    "true_force_2": 3.0,
    "true_force_3": 0.1,
    "true_force_4": 0.2,
    "true_force_5": 0.3,
    "est_contact_force_0": 1.05,
    "..._force_5": 0.32
  },
  { ... }
]
```

**Required fields per question type:**

- Q1: `timestamp_ms`, `feedback_pos_{axis}`
- Q2: `timestamp_ms`, `feedback_speed_{axis}`, `effort_current_{axis}`
- Q3: `timestamp_ms`, `vibration_0/1/2`
- Q4: `timestamp_ms`, `true_force_0/1/2` or `est_contact_force_0/1/2`
- Q5: `timestamp_ms`, `feedback_speed_{axis}`
- Q6: `timestamp_ms`, `true_torque_*` or `est_contact_torque_*`

---

## Output Format

Questions are written as JSON array:

```json
[
  {
    "id": "L1-ABB-0001",
    "question": {
      "text": "Is joint 3 at the same position in 0.0ms and 90.0ms (threshold of 0.05 rad)?",
      "level": 1,
      "type": 1
    },
    "context": {
      "episode": "data/normalized_episodes/dummy/ABB.json",
      "time_window": [0.0, 90.0],
      "joint": 3,
      "template_index": 0
    },
    "params": {
      "eps_1": 0.05,
      "eps_2": 15.0,
      "delta_1": 500,
      "eps_3": 1.0
    },
    "answer": "Yes",
    "reasoning": "I define t1=0.0ms and t2=90.0ms. I computed the joint position q(t) for joint 3. At t1 I used an exact value q(t1)=0.290000 rad, and at t2 I used an exact value q(t2)=0.380000 rad. The displacement is delta_q=0.090000 rad, which is greater than epsilon_1=0.050000 rad, so the answer is Yes.",
    "provenance": "deterministic"
  },
  { ... }
]
```

**Fields:**

- `id`: Unique identifier (episode_stem + question number)
- `question.text`: Rendered question template
- `question.level`: Always 1
- `question.type`: 1-6 (Q1-Q6)
- `context.time_window`: [t1, t2] in milliseconds
- `context.joint`: Axis/joint index 0-5 (or None for Q3/Q4)
- `context.template_index`: 0-5 (maps to question type)
- `params`: All threshold parameters used
- `answer`: Result (Yes/No for binary, scalar/vector for continuous, Unknown on error)
- `reasoning`: Plain-English derivation with intermediate values and temporal variable definitions
- `provenance`: Always "deterministic"

---

## Reasoning Trace Format

All reasoning traces follow this structure:

1. **Define temporal variables** — "I define t1={t1_ms}ms and t2={t2_ms}ms" or "I define t={t}ms"
2. **State computation steps** — "I computed X by doing Y with values Z"
3. **Show intermediate values** — Numerical values for all key computations
4. **Compare to threshold** — "X is [greater/less] than threshold, so answer is [Yes/No]"

**Example (Q2):**

```
I define t1=0.0ms and t2=90.0ms. I built symmetric windows around t1 and t2 and clipped them to the data range.
W1=[0.0,90.0] ms (raw [-500.0,500.0]) and W2=[0.0,90.0] ms (raw [-410.0,590.0]).
I used joint speed and motor current and used all samples in each window.
The friction proxies are f1=0.450000 and f2=0.475000 (median of absolute current divided by absolute speed).
The ratio f2/f1=1.055556, and the threshold is 1 plus epsilon_2 divided by 100, which is 1.150000 for epsilon_2=15 percent.
Since the ratio is not greater than the threshold, the answer is No.
```

**Key convention:** Temporal variables (t1, t2, t) are **defined once at the start**, then referenced by name throughout—not repeated as "time={value}".

---

## Reproducibility

- **Seeding:** Use `--seed N` to get deterministic question selection and rendering
- **Same episode → Same questions:** Given seed S and thresholds, questions are identical across runs
- **Answer computation:** Fully deterministic (no randomness in answer derivation)

Example:

```bash
# Run 1
python -m src.questions.cli level1 --input data.json --output q1.json --n 10 --seed 42

# Run 2 (identical output)
python -m src.questions.cli level1 --input data.json --output q2.json --n 10 --seed 42
```

---

## Testing

Test suite is in `tests/` directory (not yet included, but template:)

```bash
# Run all tests
pytest tests/test_level1.py -v

# Test Q1 position check
pytest tests/test_level1.py::test_q1_position_check -v

# Test Q5 jerk with mock data
pytest tests/test_level1.py::test_q5_jerk_mock_data -v
```

**Expected outcomes:**

- Position displacement computed correctly
- Friction proxy uses correct window clipping
- Jerk estimation handles edge cases (insufficient samples)
- Reasoning traces include all temporal variable definitions

---

## Common Issues

### "Answer computation returned no result, so the answer is Unknown."

- Indicates an exception in the answer function
- Check that input episode has required signal columns for the question type
- Verify timestamps are numeric and non-negative

### Empty reasoning trace

- Occurs if `result.get("reasoning")` returns None
- All 6 answer functions now guarantee non-empty reasoning with proper temporal variable definitions

### All Q5 answers are "Unknown"

- Requires at least 5 timestamped samples with 2 samples on each side of target time
- Dummy datasets may be too sparse; test with larger real episode

### Q2 friction always "No"

- Check that `delta_1` (window half-width) is appropriate for time scale
- If windows are too small, may not capture enough variation
- Increase `--delta-1 500` to 1000 ms or larger

---

## Integration with Main Pipeline

Questions generated here feed into the **evaluation stage**:

```
Normalized Episodes
       ↓
Level 1 Generator (this module)
       ↓
Questions JSON (150 samples per type × N episodes)
       ↓
LLM Inference (Azure OpenAI or mock adapter)
       ↓
Scoring (LLM-Match, later)
       ↓
Leaderboard
```

The `factorybench/eval/runner.py` module calls `generate_level1_questions()` to seed evaluation runs.

---

## Future Enhancements

- [ ] Batched generation for multiple episodes
- [ ] Q2 temporal analysis (detect degradation timeline)
- [ ] Integration with Isaac Sim for ground truth validation
- [ ] Performance profiling on large episodes (>100k samples)
- [ ] Caching of interpolation results

---

## References

- Main README: `../../README.md` (overview + scaling strategy)
- Paper draft: `../../docs/FactoryBench_NeurIPS_Paper_Draft.md`
- Data format spec: `../../factorybench/data/loader_tl.py`

---

**Last updated:** February 2026
**Status:** Complete (Level 1 generator stable, all 6 question types tested)
