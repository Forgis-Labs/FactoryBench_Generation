# HF dataset-viewer fix (rebuttal, addresses dataset-viewer failure)

The reviewer flagged that the HF viewer is broken for `FactoryBench/FactoryBench`.
The most common cause is a missing or incomplete `configs:` block in the
dataset-card YAML frontmatter: without it, the viewer's auto-detector
struggles with heterogeneous JSONL shards produced by our streaming
generator.

Below is a drop-in README.md YAML frontmatter block for the HF dataset
repository (`FactoryBench/FactoryBench`). It:

1. Declares one config per (source track, level) so the viewer knows which
   files to load together.
2. Pins the license to CC BY 4.0 (matches paper, matches Croissant,
   matches this repo).
3. Names each config's split and its file-glob so `datasets` can load them
   without a custom loader.
4. Adds `pretty_name`, `size_categories`, and `task_categories` for search.

Replace the frontmatter of the current `README.md` in the HF repo with this
block, keeping the body prose as-is.

```yaml
---
license: cc-by-4.0
pretty_name: FactoryBench
size_categories:
  - 10K<n<100K
task_categories:
  - question-answering
  - multiple-choice
  - time-series-forecasting
language:
  - en
tags:
  - industrial
  - robotics
  - time-series
  - causal-reasoning
  - benchmark
  - counterfactual
configs:
  - config_name: factorywave_ur3_level1
    data_files:
      - split: test
        path: factorywave_ur3_qa/level_1/test.jsonl
      - split: train
        path: factorywave_ur3_qa/level_1/train.jsonl
      - split: validation
        path: factorywave_ur3_qa/level_1/validation.jsonl
  - config_name: factorywave_ur3_level2
    data_files:
      - split: test
        path: factorywave_ur3_qa/level_2/test.jsonl
      - split: train
        path: factorywave_ur3_qa/level_2/train.jsonl
      - split: validation
        path: factorywave_ur3_qa/level_2/validation.jsonl
  - config_name: factorywave_ur3_level3
    data_files:
      - split: test
        path: factorywave_ur3_qa/level_3/test.jsonl
      - split: train
        path: factorywave_ur3_qa/level_3/train.jsonl
      - split: validation
        path: factorywave_ur3_qa/level_3/validation.jsonl
  - config_name: factorywave_ur3_level4
    data_files:
      - split: test
        path: factorywave_ur3_qa/level_4/test.jsonl
      - split: train
        path: factorywave_ur3_qa/level_4/train.jsonl
      - split: validation
        path: factorywave_ur3_qa/level_4/validation.jsonl
  - config_name: factorywave_kuka_level1
    data_files:
      - split: test
        path: factorywave_kuka_qa_260/level_1/test.jsonl
  - config_name: factorywave_kuka_level2
    data_files:
      - split: test
        path: factorywave_kuka_qa_260/level_2/test.jsonl
  - config_name: factorywave_kuka_level3
    data_files:
      - split: test
        path: factorywave_kuka_qa_260/level_3/test.jsonl
  - config_name: factorywave_kuka_level4
    data_files:
      - split: test
        path: factorywave_kuka_qa_260/level_4/test.jsonl
---
```

## If the viewer still fails after this change

1. **Shard schema drift.** Some streaming shards under
   `factorywave_kuka_qa_260/level_<N>/level<N>_shard_*.jsonl` may carry
   optional fields (e.g. `provenance.relevance`, `context.acronym_mapping`)
   that later shards drop. Run
   `python -m scripts.rebuttal.consolidate_kuka_shards --upload` first: it
   validates every record and rewrites a canonical `test.jsonl` per level,
   so the viewer sees one file with one schema.

2. **Options column type.** The `options` field is a `dict[str, str]` for
   MCQ items and `{}` for tensor / ranking items. HF's schema inference
   sometimes trips on this. Wrap the empty case as `null` at consolidation
   time by adding `--nullify-empty-options` to the consolidator (already
   supported as of the rebuttal patch), or add
   `features: { options: { dtype: struct[A: string, B: string, ...] } }`
   to each MCQ config manually.

3. **Time-series length.** The `context.time_series` field is a list of
   strings whose length varies per item. Declaring it as
   `sequence(string)` in the features spec removes the auto-inferred
   fixed-length assumption.
