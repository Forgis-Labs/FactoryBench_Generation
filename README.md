# FactoryBench: Evaluating Industrial Machine Understanding

[![Website](https://img.shields.io/badge/Website-forgis.com-orange)](https://www.forgis.com)
[![arXiv](https://img.shields.io/badge/arXiv-2605.07675-b31b1b)](https://arxiv.org/abs/2605.07675)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/FactoryBench/FactoryBench)

**Team:** Yanis Merzouki, Coral Izquierdo, Matei Ignuta-Ciuncanu, Marcos Gomez-Bracamonte, Riccardo Maggioni, Alessandro Lombardi, Camilla Mazzoleni, Federico Martelli, Balazs Gunther, Jonas Petersen, Philipp Petersen

We introduce FactoryBench, a benchmark for evaluating time-series models and LLMs on machine understanding over industrial robotic telemetry. Q&A pairs are organized along four causal levels (state, intervention, counterfactual, decision) instantiating Pearl's ladder of causation, and span five answer formats: four structured formats are scored deterministically and free-form answers are scored by an LLM-as-judge voting protocol. We propose a scalable Q&A generation framework built around structured question templates, present FactoryWave (a dense, multitask, multivariate sensor dataset collected from a UR3 cobot and a KUKA KR10 industrial arm), and construct FactoryBench as a large-scale benchmark of over 70k Q&A items grounded in roughly 15k normalized episodes from FactoryWave, AURSAD, and voraus-AD. Zero-shot evaluation of six frontier LLMs shows that no model exceeds 50% on structured levels or 18% on decision-making, revealing a wide gap between current models and operational machine understanding.

<p align="center">
  <img src="assets/factorybench_pipeline.png" width="100%">
</p>

## 4-Level Q&A Framework

| Level | Task | Example |
|-------|------|---------|
| **L1 State** | Interpret what the machine is doing now | "What is the current of joint 3 right now?" |
| **L2 Intervention** | Reason about what an action now would change | "If force in joint 3 increases to X now, what happens?" |
| **L3 Counterfactual** | Reason about what a different past would have produced | "If force had increased to X at t=20ms, what would have happened?" |
| **L4 Decision** | Generate a remediation plan from the trace | "Robot stopped with error C203A. What to do?" |

Each level builds on the previous; failure at level N implies failure at level N+1.

## FactoryWave Dataset

<p align="center">
  <img src="assets/factorybench_collage.png" width="100%">
</p>

| | |
|---|---|
| **Q&A items** | 70,000+ |
| **Episodes** | ~15,000 normalized |
| **Question templates** | 21 structured templates |
| **Answer formats** | Multi-select, scalar, tensor, ranking, free-form |
| **Robots** | UR3 cobot + KUKA KR10 |
| **Faults** | 27 types across 3 tasks |

## Quick Start

```bash
pip install datasets
```

```python
from datasets import load_dataset

ds = load_dataset("FactoryBench/FactoryBench")
```

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

Copyright (c) 2026, Forgis Labs. All rights reserved. Licensed under the [Forgis Source Code License (Non-Commercial)](LICENSE.txt).
