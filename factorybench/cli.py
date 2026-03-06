import json
import click

from .data.loader_tl import load_telemetry_literacy
from .adapters.mock import MockAdapter
from .adapters.azure_openai import AzureOpenAIAdapter
from .eval.runner import run_telemetry_literacy
from .config import AZURE_OPENAI_API_KEY


@click.group()
def cli():
    """FactoryBench CLI"""


@cli.command("run-stage1")
@click.option("--model", default="mock", help="mock | azure:<deployment>")
@click.option("--dataset-source", default="local", type=click.Choice(["local", "hf"]))
@click.option("--hf-slug", default=None)
@click.option("--hf-split", default="train")
@click.option("--fixture-path", default="datasets/stage1.json")
@click.option("--dataset-id", required=True, help="Dataset id from registry (e.g. local_basic, local_step_functions, local_patterns, hf_factoryset)")
@click.option("--limit", default=10, type=int)
def run_stage1(model, dataset_source, hf_slug, hf_split, fixture_path, dataset_id, limit):
    """Evaluate telemetry_literacy (Stage 1) and write a run JSON."""
    # Validate dataset id against registry
    from .config import DATASETS
    valid_ids = {d["id"] for d in DATASETS.get("telemetry_literacy", [])}
    if dataset_id not in valid_ids:
        raise click.UsageError(f"Invalid dataset_id '{dataset_id}'. Valid ids: {', '.join(sorted(valid_ids))}")
    
    samples = load_telemetry_literacy(
        source=dataset_source,
        path=fixture_path,
        hf_slug=hf_slug,
        split=hf_split,
        limit=limit,
    )
    if model == "mock":
        adapter, model_name = MockAdapter(), "mock"
    elif model.startswith("azure:"):
        deployment = model.split(":", 1)[1]
        if not AZURE_OPENAI_API_KEY:
            raise click.UsageError("AZURE_OPENAI_API_KEY not configured; use model=mock or set key")
        adapter, model_name = AzureOpenAIAdapter(deployment=deployment), f"azure:{deployment}"
    else:
        raise click.UsageError("Unknown model; use model=mock or azure:<deployment>")

    run = run_telemetry_literacy(
        samples=samples,
        adapter=adapter,
        model_name=model_name,
        dataset_meta={
            "source": dataset_source,
            "dataset_id": dataset_id,
            "hf_slug": hf_slug,
            "split": hf_split,
            "limit": limit,
            "fixture_path": fixture_path,
        },
    )
    click.echo(json.dumps({"run_id": run["run_id"], "aggregate": run["aggregate"]}, indent=2))


@cli.command("components:test")
@click.option("--time-series-encoder", default="default")
@click.option("--limit", default=5, type=int)
def components_test(time_series_encoder, limit):
    """Placeholder for future component tests; currently runs mock Stage 1."""
    click.echo(
        "components:test is a placeholder; running mock Stage 1 to verify plumbing..."
    )
    samples = load_telemetry_literacy(limit=limit)
    run = run_telemetry_literacy(
        samples=samples,
        adapter=MockAdapter(),
        model_name="mock",
        dataset_meta={"source": "local", "limit": limit, "fixture_path": "datasets/stage1.json"},
    )
    click.echo(json.dumps({"run_id": run["run_id"], "aggregate": run["aggregate"]}, indent=2))


if __name__ == "__main__":
    cli()
