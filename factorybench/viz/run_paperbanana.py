"""
Script to generate FactoryBench conceptual diagrams using PaperBanana.
Loads inputs from JSON files in the prompts/ directory.
"""

import asyncio
import os
import sys
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Add paperbanana to sys.path
SCRIPT_DIR = Path(__file__).parent
PAPERBANANA_PATH = SCRIPT_DIR / "paperbanana"
sys.path.append(str(PAPERBANANA_PATH))

from paperbanana import PaperBananaPipeline, GenerationInput, DiagramType
from paperbanana.core.config import Settings

# Load environment variables
load_dotenv()

# Define Output directory
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)
PROMPTS_DIR = SCRIPT_DIR / "prompts"

def build_prompt_from_json(config: dict) -> str:
    """Flatten a structured JSON prompt into a detailed string for PaperBanana."""
    # Start with basic caption and context
    caption = config.get("caption", "")
    context = config.get("context", "")
    layout = config.get("layout", "")
    
    prompt = [
        f"CAPTION: {caption}",
        f"CONTEXT: {context}",
        f"LAYOUT INSTRUCTION: {layout}"
    ]
    
    # Process boxes
    if "boxes" in config:
        prompt.append("DIAGRAM COMPONENTS:")
        for box in config["boxes"]:
            pos = box.get("position", "")
            label = box.get("label", "").replace("\n", " ")
            icon = box.get("icon", "")
            subtitle = box.get("subtitle", "").replace("\n", " ")
            color = box.get("color", "")
            prompt.append(f"- Box {pos}: Label='{label}', Icon={icon}, Subtitle='{subtitle}', Style={color}")
            
    # Process callouts
    if "callout_above" in config:
        ca = config["callout_above"]
        prompt.append(f"CALLOUT ABOVE: Position={ca.get('position')}, Style={ca.get('style')}, Header='{ca.get('header_label')}', EXACT TEXT: '{ca.get('exact_text').replace(chr(10), ' ')}'")
        
    if "callout_below" in config:
        cb = config["callout_below"]
        prompt.append(f"CALLOUT BELOW: Position={cb.get('position')}, Style={cb.get('style')}, EXACT TEXT: '{cb.get('exact_text').replace(chr(10), ' ')}'")
        
    # Process style
    if "style" in config:
        s = config["style"]
        prompt.append("GENERAL STYLE:")
        prompt.append(f"- Background: {s.get('background')}")
        prompt.append(f"- Font: {s.get('font')}")
        prompt.append(f"- Arrows: {s.get('arrows')}")
        prompt.append(f"- Palette: {', '.join(s.get('color_palette', []))}")
        prompt.append(f"- Constraints: No gradients={s.get('no_gradients')}, No 3D={s.get('no_3d_effects')}")
        prompt.append(f"- Quality: {s.get('quality')}")
        prompt.append(f"CRITICAL: {s.get('do_not_hallucinate_text')}")
        
    if "aspect_ratio" in config:
        prompt.append(f"ASPECT RATIO: {config['aspect_ratio']}")
        
    return "\n".join(prompt)

async def run_paperbanana_job(config_file: Path):
    """Run a single PaperBanana job from a JSON config file."""
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    name = config.get("name", config_file.stem)
    d_type_str = config.get("diagram_type", "methodology").upper()
    d_type = DiagramType[d_type_str]
    
    print(f"--- Generating {name} via PaperBanana ---")
    
    # Build detailed prompt
    detailed_prompt = build_prompt_from_json(config)
    
    # Configure PaperBanana for Azure OpenAI
    # Settings automatically loads from .env, using our new fallback logic for deployments
    settings = Settings(
        vlm_provider="azure_openai",
        image_provider="azure_openai_imagen",
        guidelines_path=str(PAPERBANANA_PATH / "data" / "guidelines"),
        reference_set_path=str(PAPERBANANA_PATH / "data" / "reference_sets"),
        output_dir=str(OUTPUT_DIR / "paperbanana_runs"),
        optimize_inputs=True,
        auto_refine=True,
    )
    
    pipeline = PaperBananaPipeline(settings=settings)
    
    # Run pipeline with the detailed prompt as communicative_intent
    result = await pipeline.generate(
        GenerationInput(
            source_context=config.get("context", ""),
            communicative_intent=detailed_prompt,
            diagram_type=d_type,
        )
    )
    
    # Copy final result to figures directory
    final_path = OUTPUT_DIR / f"{name}.png"
    shutil.copy(result.image_path, final_path)
    print(f"Saved {name} to {final_path}")

async def main():
    if len(sys.argv) > 1:
        # Run specific figure(s) if provided
        for arg in sys.argv[1:]:
            config_file = PROMPTS_DIR / f"{arg}.json"
            if config_file.exists():
                await run_paperbanana_job(config_file)
            else:
                print(f"Error: Prompt file {config_file} not found.")
    else:
        # Run all figures in the prompts directory
        print("Starting batch generation for all prompts...")
        prompt_files = sorted(PROMPTS_DIR.glob("*.json"))
        for config_file in prompt_files:
            await run_paperbanana_job(config_file)

if __name__ == "__main__":
    asyncio.run(main())
