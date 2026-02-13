# QA Generation

This directory contains notebooks and scripts to generate Q&A pairs for FactoryBench.

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)

## Setup Environment

1. **Create a new Conda environment:**

   ```bash
   conda create -n qa_gen python=3.10
   ```

2. **Activate the environment:**

   ```bash
   conda activate qa_gen
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Notebooks

1. **Open `generate_qa.ipynb`** (or `generate_qa rag.ipynb`) and select the "QA Generation" kernel.

## Configuration

- Ensure you have a `.env` file in the `FactoryBench` root directory if you are using the RAG notebook, with the following keys:
  - `AZURE_OPENAI_KEY`
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_DEPLOYMENT`
