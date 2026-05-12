import os
import json
from datetime import date
from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
import pyarrow.parquet as pq

load_dotenv()

DATASET_NAME = "FactoryBench/FactoryBench"
DATA_ROOT = "factorybench_qa"
WAVE_ROOT = "factorywave"
HF_TOKEN = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)
hf_fs = HfFileSystem(token=HF_TOKEN)

# Discover levels by listing repo files
all_files = list(api.list_repo_files(DATASET_NAME, repo_type="dataset"))
data_files = [f for f in all_files if f.startswith(f"{DATA_ROOT}/") and f.endswith(".jsonl")]
wave_files = [f for f in all_files if f.startswith(f"{WAVE_ROOT}/") and f.endswith(".parquet")]

# Collect one sample file per level subfolder
level_samples: dict = {}
for path in data_files:
    parts = path.split("/")
    if len(parts) >= 3:
        level_dir = parts[1]  # e.g. "level_1", "level_2", ...
        if level_dir not in level_samples:
            level_samples[level_dir] = path

if not level_samples:
    raise RuntimeError("No level subfolders found in the dataset repository.")


def infer_type(value) -> str:
    if isinstance(value, bool):
        return "sc:Boolean"
    if isinstance(value, int):
        return "sc:Integer"
    if isinstance(value, float):
        return "sc:Float"
    if isinstance(value, list):
        return "sc:ItemList"
    if isinstance(value, dict):
        return "sc:StructuredValue"
    return "sc:Text"


def build_fields(sample_item: dict) -> list:
    return [
        {
            "@type": "cr:Field",
            "name": name,
            "dataType": infer_type(value),
        }
        for name, value in sample_item.items()
    ]


def arrow_to_sc(arrow_type) -> str:
    import pyarrow as pa
    if pa.types.is_boolean(arrow_type):
        return "sc:Boolean"
    if pa.types.is_integer(arrow_type):
        return "sc:Integer"
    if pa.types.is_floating(arrow_type):
        return "sc:Float"
    if pa.types.is_timestamp(arrow_type) or pa.types.is_date(arrow_type):
        return "sc:DateTime"
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return "sc:ItemList"
    if pa.types.is_struct(arrow_type) or pa.types.is_map(arrow_type):
        return "sc:StructuredValue"
    return "sc:Text"


def build_parquet_fields(arrow_schema) -> list:
    return [
        {"@type": "cr:Field", "name": field.name, "dataType": arrow_to_sc(field.type)}
        for field in arrow_schema
    ]


# Build one RecordSet per level using a downloaded sample to infer the schema
record_sets = []
for level_dir, sample_path in sorted(level_samples.items()):
    local_path = hf_hub_download(
        DATASET_NAME,
        sample_path,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    with open(local_path, encoding="utf-8") as f:
        first_line = f.readline()
    sample_item = json.loads(first_line)

    level_file_count = sum(1 for p in data_files if p.split("/")[1] == level_dir)

    record_sets.append({
        "@type": "cr:RecordSet",
        "name": level_dir,
        "description": f"Q&A pairs for {level_dir.replace('_', ' ')} of FactoryBench.",
        "cr:source": {
            "@type": "cr:FileSet",
            "containedIn": f"https://huggingface.co/datasets/{DATASET_NAME}/tree/main/{DATA_ROOT}/{level_dir}",
            "encodingFormat": "application/x-jsonlines",
            "cr:fileCount": level_file_count,
        },
        "field": build_fields(sample_item),
    })


# Build one RecordSet per factorywave parquet, reading the schema remotely (footer only)
WAVE_DESCRIPTIONS = {
    "episodes.parquet": "Episode-level metadata (robot, fault, counterfactual links).",
    "flow.parquet": "Task flow definitions (normal vs counter_factual flows).",
    "kuka_signals.parquet": "KUKA KR10 industrial-arm telemetry signals at ~83 Hz nominal.",
    "ur_signals.parquet": "UR3 cobot telemetry signals at ~125 Hz nominal (3,076 episodes).",
    "ur_signals_10hz.parquet": "UR3 cobot telemetry signals downsampled to 10 Hz (3,984 episodes; disjoint from ur_signals.parquet).",
    "ur_screwdriver_signals.parquet": "UR3 cobot telemetry signals for screwdriver-task episodes at ~125 Hz nominal.",
}

for wave_path in sorted(wave_files):
    fname = wave_path.split("/")[-1]
    record_name = fname.removesuffix(".parquet")

    schema = pq.read_schema(f"datasets/{DATASET_NAME}/{wave_path}", filesystem=hf_fs)
    description = WAVE_DESCRIPTIONS.get(fname, f"FactoryWave file {fname}.")

    record_sets.append({
        "@type": "cr:RecordSet",
        "name": record_name,
        "description": description,
        "cr:source": {
            "@type": "cr:FileObject",
            "contentUrl": f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main/{wave_path}",
            "encodingFormat": "application/vnd.apache.parquet",
        },
        "field": build_parquet_fields(schema),
    })

croissant = {
    "@context": {
        "@language": "en",
        "@vocab": "https://schema.org/",
        "sc": "https://schema.org/",
        "cr": "https://mlcommons.org/croissant/",
        "dct": "http://purl.org/dc/terms/",
    },
    "@type": "sc:Dataset",
    "cr:conformsTo": "http://mlcommons.org/croissant/1.0",
    "name": "FactoryBench",
    "description": (
        "FactoryBench is a benchmark for evaluating machine-behavior reasoning "
        "over industrial robotic telemetry, structured around Pearl's causal "
        "hierarchy. Q&A pairs span four reasoning levels: state identification, "
        "intervention reasoning, counterfactual analysis, and engineering "
        "decision-making (troubleshooting and optimisation). The benchmark is "
        "grounded in FactoryWave, a dense multivariate telemetry dataset "
        "collected from a UR3 cobot (~125 Hz) and a KUKA KR10 industrial arm "
        "(~83 Hz) with systematic fault injection."
    ),
    "url": f"https://huggingface.co/datasets/{DATASET_NAME}",
    "license": "https://opensource.org/licenses/MIT",
    "version": "1.0",
    "citation": (
        "Merzouki et al., FactoryBench: A Benchmark for Machine-Behavior Reasoning "
        "over Industrial Robotic Telemetry, NeurIPS 2026."
    ),
    "datePublished": str(date.today()),
    "cr:recordSet": record_sets,
    "cr:rai": {
        "rai:dataUseCases": (
            "Benchmark evaluation of LLMs and time-series models on structured "
            "industrial Q&A reasoning tasks (state, intervention, counterfactual, "
            "decision-making)."
        ),
        "rai:dataLimitations": (
            "Domain-specific to factory and industrial robotic scenarios; "
            "may not generalise to open-domain Q&A tasks. Faults are atomic "
            "and drawn from a closed catalogue of 27 physically injected "
            "mechanisms, which differs from compound or gradual real-world faults."
        ),
        "rai:dataBiases": (
            "Potential domain bias toward industrial workflows and UR/KUKA "
            "robot terminology; limited diversity of robot platforms and tasks. "
            "Size imbalance between levels 2 and 3."
        ),
        "rai:personalSensitiveInformation": (
            "The dataset contains no personal or sensitive information. "
            "All data was collected from industrial robots with no human subjects involved."
        ),
        "rai:dataSocialImpact": (
            "Advances machine-behavior reasoning research for industrial robotics, "
            "supporting safer and more reliable automated manufacturing systems. "
            "Not intended for deployment in safety-critical systems without further expert validation."
        ),
        "rai:hasSyntheticData": (
            "Yes. Q&A pairs are synthetically generated from real robotic telemetry "
            "signals using templated question generation. The underlying sensor data "
            "is collected from physical UR3 and KUKA KR10 robots executing "
            "pick-and-place, screwing, and peg-in-hole tasks at 83-125 Hz "
            "with systematic fault injection, supplemented with AURSAD and voraus-AD datasets."
        ),
    },
}

output_file = "factorybench_croissant.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(croissant, f, indent=2, ensure_ascii=False)

print(f"Croissant file saved to: {output_file}")
print(f"Record sets: {[rs['name'] for rs in record_sets]}")
