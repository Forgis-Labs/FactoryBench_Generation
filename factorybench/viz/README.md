# Figure Generation with PaperBanana

- **Conceptual diagrams** — [PaperBanana](https://github.com/llmsresearch/paperbanana) + Azure FLUX.2-pro: pipeline diagrams, architecture overviews


## Clone PaperBanana (Edited fork)

From the **FactoryBench root**:

```powershell
git clone -b fix/update-deprecated-gemini-model https://github.com/coral2742/paperbanana.git factorybench\viz\paperbanana
```

## Install dependencies

With the virtual environment activated:

```powershell
python -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -e "factorybench\viz\paperbanana[openai]"
pip install matplotlib numpy pandas python-dotenv
```

## Configure `.env`

Create a `.env` in the **project root**. Ask a team member for the key values:

```env
# Azure OpenAI — VLM (planning & critique)
VLM_PROVIDER=openai
OPENAI_API_KEY=<your-azure-openai-api-key>
OPENAI_BASE_URL=<your-azure-openai-base-url>
OPENAI_VLM_MODEL=<your-azure-openai-model>
AZURE_OPENAI_API_VERSION=2024-02-01

# Azure AI Foundry — FLUX.2-pro (image generation)
IMAGE_PROVIDER=openai_imagen
FLUX_API_KEY=<your-flux-api-key>
FLUX_BASE_URL=<your-flux-base-url>
OPENAI_IMAGE_MODEL=<your-flux-model>
```


## Usage

### Conceptual Diagrams (PaperBanana)

```powershell
cd factorybench\viz\paperbanana

# Generate a single diagram
python -m paperbanana.cli generate --input examples/sample_inputs/transformer_method.txt --caption "Your caption"

# example
python -m paperbanana.cli generate --input examples/sample_inputs/transformer_method.txt --caption "Overview of our encoder-decoder architecture with sparse routing"

```

Output: `figures/<name>.png`

---

## Customizing Prompts

Add `.txt` files to `prompts/`. Each file is passed directly as the `--input` argument. See existing examples:

| File | Description |
|------|-------------|
| `factorybench_method.txt` | Methodology context in bullet-point format |
| `factorybench_method_paper_level.txt` | Methodology context in paper-level prose |
| `factorybench_neurips.txt` | Detailed zone-by-zone layout with hex colors and style rules |



> See `agent.md` for the full FactoryBench color palette and level color mapping.

## References

- [PaperBanana](https://github.com/llmsresearch/paperbanana) — [paper](https://arxiv.org/abs/2601.23265)