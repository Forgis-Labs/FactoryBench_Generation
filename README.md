# FactoryBench: Evaluating Industrial Machine Understanding

[![Website](https://img.shields.io/badge/Website-forgis.com-orange)](https://www.forgis.com)
[![arXiv](https://img.shields.io/badge/arXiv-2605.07675-b31b1b)](https://arxiv.org/abs/2605.07675)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/FactoryBench/FactoryBench)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**Team:** Yanis Merzouki, Coral Izquierdo, Matei Ignuta-Ciuncanu, Marcos Gomez-Bracamonte, Riccardo Maggioni, Alessandro Lombardi, Camilla Mazzoleni, Federico Martelli, Balazs Gunther, Jonas Petersen, Philipp Petersen

This repository contains the Q&A generation framework, evaluation drivers, and scoring pipeline for FactoryBench. For the lightweight Python client (pip-installable), see [Forgis-Labs/FactoryBench](https://github.com/Forgis-Labs/FactoryBench).

We introduce FactoryBench, a benchmark for evaluating time-series models and LLMs on machine understanding over industrial robotic telemetry. Q&A pairs are organized along four causal levels (state, intervention, counterfactual, decision) instantiating Pearl's ladder of causation, and span five answer formats: four structured formats are scored deterministically and free-form answers are scored by an LLM-as-judge voting protocol. We propose a scalable Q&A generation framework built around structured question templates, present FactoryWave (a dense, multitask, multivariate sensor dataset collected from a UR3 cobot and a KUKA KR10 industrial arm), and construct FactoryBench as a large-scale benchmark of over 70k Q&A items grounded in roughly 15k normalized episodes from FactoryWave, AURSAD, and voraus-AD. Zero-shot evaluation of six frontier LLMs shows that no model exceeds 50% on structured levels or 18% on decision-making, revealing a wide gap between current models and operational machine understanding.

## 4-Level Q&A Framework

| Level | Task | Example |
|-------|------|---------|
| **L1 State** | Interpret what the machine is doing now | "What is the current of joint 3 right now?" |
| **L2 Intervention** | Reason about what an action now would change | "If force in joint 3 increases to X now, what happens?" |
| **L3 Counterfactual** | Reason about what a different past would have produced | "If force had increased to X at t=20ms, what would have happened?" |
| **L4 Decision** | Generate a remediation plan from the trace | "Robot stopped with error C203A. What to do?" |

Each level builds on the previous; failure at level N implies failure at level N+1.

## Quick Start

```bash
git clone https://github.com/Forgis-Labs/FactoryBench_Generation.git
cd FactoryBench_Generation
pip install -e .
```

### Load the dataset

```python
from datasets import load_dataset

ds = load_dataset("FactoryBench/FactoryBench", split="test")
```

### Run evaluation

```bash
python -m src.evaluation.run_foundry_eval \
    --model gpt-5-mini \
    --level 1,2,3,4 \
    --output-dir runs/gpt5mini-fbench
```

### Score predictions

```bash
python -m src.scoring.score_traces \
    --predictions runs/gpt5mini-fbench \
    --judge-model gpt-5-mini
```

## Scoring

| Format | Score definition |
|--------|-----------------|
| Multiple choice (single) | 1 if extracted letter matches ground truth, else 0 |
| Multiple choice (multi, T/F) | Fraction of positions matching ground truth |
| Ranking | Kendall-tau: 1 = exact match, 0.5 = random, 0 = reversed |
| Numerical | 1 within margin, 0.5 within 2x margin, 0 otherwise |
| Tensor | Per-channel margin, averaged |
| Free-form | LLM-judge rubric: 0 / 0.5 / 1 |

## Citation

```bibtex
@article{merzouki2026factorybench,
  title   = {FactoryBench: Evaluating Industrial Machine Understanding},
  author  = {Merzouki, Yanis and Izquierdo, Coral and Ignuta-Ciuncanu, Matei
             and Gomez-Bracamonte, Marcos and Maggioni, Riccardo and Lombardi,
             Alessandro and Mazzoleni, Camilla and Martelli, Federico and
             Gunther, Balazs and Petersen, Jonas and Petersen, Philipp},
  journal = {arXiv preprint arXiv:2605.07675},
  year    = {2026}
}
```

## License

Copyright (c) 2026, Forgis. Licensed under [CC BY-NC-SA 4.0](LICENSE).
