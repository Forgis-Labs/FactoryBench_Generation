"""Fine-tune Qwen3-4B on FactoryBench QA (levels 1-3) with Unsloth.

Single GPU, 4-bit, ~10GB VRAM. Uses TRL SFTTrainer for simplicity.

Usage:
    # Local GPU
    uv run python tmp/train_factorybench_unsloth.py

    # Custom settings
    uv run python tmp/train_factorybench_unsloth.py --batch_size 2 --epochs 1
"""

import argparse
import json
import os
import sys
from pathlib import Path

print("[BOOT] Script started", flush=True)


def load_factorybench(levels=(1, 2, 3), splits=("train", "validation", "test"),
                       data_dir=None):
    """Load FactoryBench QA data for specified levels."""
    data = {}
    for split in splits:
        samples = []
        for level in levels:
            if data_dir:
                path = os.path.join(data_dir, f"level_{level}_{split}.jsonl")
                if not os.path.exists(path):
                    print(f"  SKIP level_{level}_{split}.jsonl (not found)")
                    continue
            else:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(
                    "Forgis/FactoryBench_QA_pairs",
                    f"factorynet_qa_150k/level_{level}/{split}.jsonl",
                    repo_type="dataset",
                )
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    samples.append(row)
            print(f"  Level {level} {split}: {len(samples)} samples (cumulative)")
        data[split] = samples
    return data


def format_sample(sample, tokenizer, max_length=8192):
    """Convert a FactoryBench sample to a ChatML conversation for SFTTrainer."""
    question = sample["question"]
    context = sample.get("context", {})
    answer = str(sample.get("answer", ""))

    # Extract time series text
    ts_data = context.get("time_series", [])
    acronym_map = context.get("time_series_format", {}).get("acronym_mapping", {})

    parts = []

    # Acronym mapping
    if acronym_map:
        mapping_str = ", ".join(f"{k}={v}" for k, v in list(acronym_map.items())[:15])
        if len(acronym_map) > 15:
            mapping_str += f", ... ({len(acronym_map)} total)"
        parts.append(f"Feature mapping: {mapping_str}")

    # Time series data
    if isinstance(ts_data, list) and ts_data:
        ts_str = "\n".join(ts_data)
        parts.append(f"Time series data:\n{ts_str}")
    elif isinstance(ts_data, str):
        parts.append(f"Time series data:\n{ts_data}")

    # Provenance
    provenance = sample.get("provenance", {})
    dataset = provenance.get("dataset", "")
    episode = provenance.get("episode", "")
    if dataset:
        parts.append(f"Dataset: {dataset}, Episode: {episode}")

    # Options (MCQ)
    options = sample.get("options", {})
    if options:
        opts_str = "\n".join(f"  {k}: {v}" for k, v in options.items())
        question += f"\n\nOptions:\n{opts_str}"

    context_str = "\n\n".join(parts)
    user_content = f"{context_str}\n\n{question}" if context_str else question

    # Return as pre-formatted ChatML text
    text = (f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>")
    return {"text": text}


def main():
    p = argparse.ArgumentParser(description="Fine-tune Qwen3-4B on FactoryBench with Unsloth")
    p.add_argument("--model_name", default="unsloth/Qwen3-4B")
    p.add_argument("--output_dir", default="results/factorybench_unsloth")
    p.add_argument("--data_dir",
                   default=os.environ.get("SM_CHANNEL_DATA", "data/factorybench"),
                   help="Path to JSONL files")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FactoryBench Fine-tuning with Unsloth")
    print(f"  Model: {args.model_name}")
    print(f"  Max length: {args.max_length}")
    print(f"  Batch: {args.batch_size} x {args.grad_accum} grad_accum = {args.batch_size * args.grad_accum} effective")
    print("=" * 60)

    # Load model with Unsloth
    print("\n[1/4] Loading model...", flush=True)
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_length,
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )
    print(f"  Model loaded: {args.model_name}", flush=True)

    # Apply LoRA
    print("[2/4] Applying LoRA...", flush=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",  # Unsloth optimized
        random_state=args.seed,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Load data
    print("\n[3/4] Loading FactoryBench data...", flush=True)
    raw_data = load_factorybench(levels=(1, 2, 3), data_dir=args.data_dir)

    # Format for SFTTrainer
    from datasets import Dataset

    def format_all(samples):
        formatted = []
        for s in samples:
            fmt = format_sample(s, tokenizer, max_length=args.max_length)
            formatted.append(fmt)
        return formatted

    train_formatted = format_all(raw_data["train"])
    val_formatted = format_all(raw_data.get("validation", raw_data.get("test", [])))

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted) if val_formatted else None

    print(f"  Train: {len(train_dataset)} samples")
    if val_dataset:
        print(f"  Val: {len(val_dataset)} samples")

    # Setup trainer
    print("\n[4/4] Starting training...", flush=True)
    from trl import SFTTrainer, SFTConfig
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            output_dir=str(output_dir),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=50,
            save_steps=500,
            save_total_limit=3,
            eval_strategy="epoch" if val_dataset else "no",
            seed=args.seed,
            max_seq_length=args.max_length,
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
    )

    # Print memory before training
    import torch
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM before training: {vram:.1f} GB")

    # Train
    stats = trainer.train()

    # Save
    model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    # Summary
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"  Training loss: {stats.training_loss:.4f}")
    print(f"  Peak VRAM: {peak_vram:.1f} GB")
    print(f"  Adapter saved: {output_dir / 'adapter'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
