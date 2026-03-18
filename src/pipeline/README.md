# Pipeline for Level 1
1. Q&A generation
2. Prompt generation
3. Model answering
4. Evaluation of models
5. Results, upload to Opik and plots

### Use

```bash
python src/pipeline/run_level1_pipeline.py -n 100
```

Run test mode:
```bash
python src/pipeline/run_level1_pipeline.py -n 5 --test-mode
```

Make sure you have defined on `.env`:
```bash
AZURE_OPENAI_ENDPOINT="https://xeleritbase.openai.azure.com/"
AZURE_OPENAI_API_KEY="<your_api_key>"
AZURE_OPENAI_DEPLOYMENT="student-gpt-4.1"
HF_API_TOKEN="<your_api_key>"
AZURE_OPENAI_API_VERSION="2024-02-01"
OPIK_API_KEY="<your_api_key>"
OPIK_PROJECT_NAME="FactoryBench"
OPIK_WORKSPACE="forgis"
```
