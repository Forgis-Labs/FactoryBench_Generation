Level 1 Question Generator

This folder contains tools to generate Level 1 (state identification) questions.

Files:

- `level1.py` - generator module and CLI
- `cli.py` - top-level CLI helper
- `level1_config.json` - default parameters for question generation

Quick start (generate 5 example questions):

```bash
python -m src.questions.cli level1 \
  --input datasets/open_datasets/ur3+cobotops/normalized/ur3e_episode_001.json \
  --output datasets/open_datasets/ur3+cobotops/normalized/questions_l1_sample.json \
  --n 5 -v
```
