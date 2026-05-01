# Batch inference across all FactoryBench models

End-to-end guide for running batch inference on the four production models
(`gpt-5.1-1`, `claude-sonnet-4.6`, `mistral-large-3`, `deepseek-v3.2`) against
QA pairs that already live on Hugging Face. Optimised for the lowest cost per
prompt and minimal hand-holding.

> Pipeline orchestration, stage flow, and Q&A generation are documented in
> [src/pipeline/README.md](../src/pipeline/README.md). AWS account setup
> (S3 buckets, IAM roles, Bedrock model IDs) is in
> [src/evaluation/aws-setup.md](../src/evaluation/aws-setup.md). This doc
> covers only the **eval** stage at scale.

## TL;DR

```bash
conda activate factorybench

python -m src.pipeline.run_pipeline \
    --stages fetch,prompts,eval \
    --hf-dataset-folder factorynet_qa_150k \
    --levels 1,2,3,4 \
    --models gpt-5.1-1,claude-sonnet-4.6,mistral-large-3,deepseek-v3.2 \
    --no-judge \
    --cost-limit 5
```

This pulls QA pairs from `Forgis/FactoryBench_QA_pairs/factorynet_qa_150k/`,
builds prompts, dispatches batch jobs across all 4 models on their respective
provider backends, polls to completion, downloads results, and writes one
reply JSON per prompt under `output/replies/level{N}/{slug}/`.

## Provider matrix

| Model | Provider | API style | Batch API | Min batch size |
|---|---|---|---|---|
| `gpt-5.1-1` | Azure Foundry | OpenAI `/v1/batches` | yes | none (Azure) |
| `claude-sonnet-4.6` | AWS Bedrock | Anthropic Messages | `CreateModelInvocationJob` | **100** |
| `mistral-large-3` | AWS Bedrock | Mistral chat | `CreateModelInvocationJob` | **100** |
| `deepseek-v3.2` | AWS Bedrock | DeepSeek chat | `CreateModelInvocationJob` | **100** |

The runner picks the right backend automatically based on
[src/config.py](../src/config.py)`.MODELS`. Adding a new model is one dict
edit there.

### Why two backends

* **Azure Foundry** hosts the OpenAI / GPT-5.x family that AWS doesn't carry.
  Batch is `POST /v1/files` + `POST /v1/batches` with a JSONL of
  `chat.completions` requests, polled until `completed`.
* **AWS Bedrock** hosts everything else. Native batch via
  `CreateModelInvocationJob` reads/writes JSONL on S3, runs at ~50% of
  on-demand price, and clears in a few hours.

## Required `.env`

Combine the relevant rows from the per-stage docs. Minimal set for *batch
inference only* (no Q&A generation, no Opik):

```bash
# --- Hugging Face (to fetch QA pairs) ---
HF_API_TOKEN="<your_hf_token>"

# --- Azure Foundry (gpt-5.1-1) ---
AZURE_API_KEY="<your_azure_key>"
CHAT_ENDPOINT="https://student-research-lab-resource.services.ai.azure.com/openai/v1"
# OPTIONAL — separate deployment for batch on a globalbatch SKU. See "GPT-5.1
# batch caveat" below. Leave unset to use the default deployment for batch.
GPT_5_1_BATCH_DEPLOYMENT="gpt-5.1-batch"

# --- AWS shared (Bedrock + SageMaker) ---
AWS_PROFILE="factorybench"            # or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY
FB_S3_BUCKET="factorybench-batch-io"   # any bucket the IAM roles can read+write
FB_S3_PREFIX="factorybench/"           # default 'factorybench/'
BEDROCK_BATCH_ROLE_ARN="arn:aws:iam::<acct>:role/factorybench-bedrock-batch"

# --- Bedrock per-model id + region (regions differ by model) ---
CLAUDE_SONNET_46_MODEL_ID="eu.anthropic.claude-sonnet-4-6"   # CRIS profile
CLAUDE_SONNET_46_REGION="eu-central-1"
MISTRAL_LARGE_3_MODEL_ID="mistral.mistral-large-3-675b-instruct"
MISTRAL_LARGE_3_REGION="us-west-2"                            # no EU region yet
DEEPSEEK_V32_MODEL_ID="deepseek.v3.2"
DEEPSEEK_V32_REGION="eu-west-2"                               # or eu-north-1
```

The full `.env` template (with judge, Opik, SageMaker fields) is in
[src/pipeline/README.md](../src/pipeline/README.md).

## Common run patterns

### Bench a fresh QA folder uploaded to HF

```bash
python -m src.pipeline.run_pipeline \
    --stages fetch,prompts,eval \
    --hf-dataset-folder factorynet_qa_150k \
    --levels 1,2,3,4 \
    --models gpt-5.1-1,claude-sonnet-4.6,mistral-large-3,deepseek-v3.2 \
    --no-judge \
    --cost-limit 5
```

`--stages fetch,prompts,eval` skips generation and pulls from HF.
`--no-judge` skips the LLM-as-judge call (saves time/$ during the inference
run; you can re-judge later by re-running with judge enabled).

### Bench against a local QA folder

```bash
python -m src.pipeline.run_pipeline \
    --stages prompts,eval \
    --questions-dir output/questions/level{level} \
    --levels 1,2,3,4 \
    --models gpt-5.1-1,claude-sonnet-4.6,mistral-large-3,deepseek-v3.2 \
    --no-judge \
    --cost-limit 5
```

### Run only a subset of models or a single level

```bash
# L2 only, two models, batch enabled
python -m src.pipeline.run_pipeline \
    --stages fetch,prompts,eval \
    --hf-dataset-folder factorynet_qa_150k \
    --levels 2 \
    --models claude-sonnet-4.6,gpt-5.1-1 \
    --no-judge --cost-limit 3
```

### Force concurrent sync (no batch)

If a Bedrock batch job fails on submission validation or you need real-time
results, drop `--no-batch`-style fallback by running:

```bash
python -m src.pipeline.run_pipeline ... --no-batch --concurrency 8
```

This is rarely the right answer; batch is ~50% cheaper on Bedrock and the
only sane way to do >1k prompts. Use sync only for debugging.

## Provider-specific gotchas

### GPT-5.1 batch caveat (Azure Foundry)

Azure batch on GPT-5.1 requires a deployment whose SKU is `globalbatch`
(or `datazonebatch`). The default `GlobalStandard` SKU **rejects batch with
HTTP 400** `invalid_deployment_type`. Two ways to fix:

1. **Recommended**: keep the existing `gpt-5.1-1` deployment as
   `GlobalStandard` for sync (used by the LLM-as-judge), and create a
   *second* deployment named e.g. `gpt-5.1-batch` on the `globalbatch` SKU.
   Set `GPT_5_1_BATCH_DEPLOYMENT="gpt-5.1-batch"` in `.env`. The runner
   automatically routes batch traffic to the override and sync traffic to
   the original.
2. Or upgrade the existing deployment to `globalbatch` in place — but then
   *sync calls fail*, breaking the LLM-as-judge.

If batch ever falls back to sync, the pipeline logs:

```
WARNING: [batch] gpt-5.1-1 batch submission failed (...); falling back to concurrent sync.
```

### Bedrock 100-record minimum

`CreateModelInvocationJob` requires at least 100 records per job. Smaller
batches automatically fall back to **concurrent sync** (`bedrock-sync` in
the logs). Costs are tiny at that scale (cents) but throughput is lower.

For benchmark runs at production scale, batch always wins. If you're
running a 16-prompt subset for debugging, expect sync.

### Bedrock validation errors

If `CreateModelInvocationJob` returns a validation error (e.g. malformed
input JSONL, missing IAM permissions), the runner falls back to sync. The
pre-flight cost estimate appears before submission:

```
INFO: [bedrock-batch] claude-sonnet-4.6: pre-flight estimate ~$1.83 ...
```

If the pre-flight exceeds `--cost-limit`, the run aborts before submitting.

### LLM-as-judge cost

The default judge (`gpt-5.1-1`) runs **synchronously per reply** for every
free-form item. For 30k free-form replies this is ~$30–60 of judge cost on
top of the model inference. Recommendations:

* For benchmark runs at scale: use `--no-judge`, then re-judge later with
  `--stages eval --no-batch ... --judge-model gpt-5.1-1` once you've
  validated model results.
* For small dev runs: judge inline (default).

## Monitoring a long run

Per-model + per-level summaries land at
`output/replies/level{N}/{slug}/_summary.json` after each run. While in
flight:

```bash
# Reply counts so far
for d in output/replies/level*/*/; do
  echo "$d $(ls "$d"/*_answer.json 2>/dev/null | wc -l)"
done

# Pipeline log tails (one per launched process; we usually launch one process
# per level and let the runner fan out across models)
tail -f output/run.err.log

# Bedrock batch status (per model)
grep -E "bedrock-batch.*(submitted|status=|completed)" output/run.err.log

# Azure batch status
grep -E "gpt-5.1-1 batch=.*status=" output/run.err.log
```

A successful run prints the per-model summary at the end of each level:

```
INFO: Done. completed=100 failed=0 skipped=0
```

## Cost expectations

For 100-prompt batches with ~5k input tokens and ~1k output each:

| Model | Per-batch cost (~100 prompts) |
|---|---|
| `gpt-5.1-1` | ~$0.20–0.50 |
| `claude-sonnet-4.6` | ~$1.50–2.00 |
| `mistral-large-3` | ~$0.20 |
| `deepseek-v3.2` | ~$0.10–0.15 |

`--cost-limit` is enforced per (level × model) batch. Set conservatively;
the pre-flight estimate ratchets up with input length.

## What runs where

* [src/pipeline/run_pipeline.py](../src/pipeline/run_pipeline.py)
  orchestrates the levels and dispatches per-model eval jobs.
* [src/evaluation/run_foundry_eval.py](../src/evaluation/run_foundry_eval.py)
  handles Azure Foundry models (`gpt-5.x`, judge calls, Foundry-style
  batch).
* [src/evaluation/run_aws_eval.py](../src/evaluation/run_aws_eval.py)
  handles Bedrock + SageMaker models (Bedrock batch via S3, SageMaker async
  per-request).
* Provider routing is in [src/config.py](../src/config.py)`.MODELS`.

## Re-scoring without re-running inference

If you change scoring rules or acceptance bounds and want to re-score
existing replies (no new API calls):

```bash
# In-place re-score using current scorer logic + bound semantics. The script
# writes the updated `score` field back to each reply file.
python -m src.evaluation.per_template_scores \
    --levels 1,2,3,4 \
    --include-noised \
    --output output/per_template_scores.csv
```

This is the workflow used after every numerical/scorer fix in
[src/evaluation/run_foundry_eval.py](../src/evaluation/run_foundry_eval.py)
to avoid paying the inference cost twice.
