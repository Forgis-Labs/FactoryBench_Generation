# NeurIPS Image Generator
Generates publication-quality figures for the paper using Azure AI Foundry (MAI-Image-2), with optional AI-driven prompt refinement.

### Setup

1. Create a `.env` file with your Azure API key:
   ```
    AZURE_OPENAI_API_KEY="<your_key_here>"
    CHAT_MODEL="gpt-5-mini" 
    CHAT_ENDPOINT="<base_endpoint>/openai/v1" 
    
    REASONING_MODEL="claude-opus-4-6" 
    REASONING_ENDPOINT="<base_endpoint>/anthropic/v1" 

    IMAGE_MODEL="MAI-Image-2" 
    IMAGE_ENDPOINT="<base_endpoint>/mai/v1" 
   ```
2. Install dependencies:
   ```bash
   pip install requests python-dotenv openai
   ```

### Usage

Create a new text file describing your idea inside `\prompts` folder.

```bash
# Use a specific prompt file
python scripts/neurips_image_generator.py --prompt scripts/prompts/factorybench_method.txt

# Enable iterative refinement (generate -> critique -> improve -> regenerate)
python scripts/neurips_image_generator.py --refine --iterations 3

# Use Claude instead of GPT for refinement
python scripts/neurips_image_generator.py --refine --agent claude
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | `prompts/factorybench_neurips.txt` | Path to the prompt file |
| `--outdir` | `outputs/run_<timestamp>` | Output directory |
| `--width` | 1366 | Image width in pixels |
| `--height` | 768 | Image height in pixels |
| `--refine` | off | Enable AI-driven prompt refinement |
| `--agent` | `gpt` | Refinement model: `gpt` or `claude` |
| `--iterations` | 3 | Number of generate-refine cycles (with `--refine`) |

### Output

Each run creates a timestamped folder in `outputs/` containing:
- `diagram_iter_N.png` -- generated images
- `prompt_iter_N.txt` -- prompt used per iteration
- `critique_iter_N.txt` -- AI critique (when using `--refine`)
- `metadata.json` -- run config and cost summary
