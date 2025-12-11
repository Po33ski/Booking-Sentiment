from __future__ import annotations

from typing import Optional
import typer

from .config import ProjectConfig
from .runtime import configure_runtime
from .cli_functions import (
    run_load,
    run_clean,
    run_quality,
    run_split,
    run_fine_tune,
    run_evaluate,
    run_inference,
    run_explain,
    run_scan,
    run_purge,
)
# app: root of the CLI
app = typer.Typer(add_completion=False, help="Booking Sentiment - MLOps-friendly CLI")


# Shared helper to hydrate config and apply deterministic settings
def _load_config(config_path: Optional[str]) -> ProjectConfig:
    cfg = ProjectConfig.load(config_path)
    configure_runtime(cfg.fine_tune.seed, cfg.fine_tune.device)
    return cfg


# load: load the raw dataset from HuggingFace and return positive/negative Series
@app.command()
def load(config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to JSON config file")) -> None:
    cfg = _load_config(config)
    run_load(cfg)

# clean: clean the raw dataset and return a dataframe with columns: text, label
@app.command()
def clean(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = _load_config(config)
    run_clean(cfg)

# quality: run the quality analysis (CleanLab) and return a dataframe with columns: text, label
@app.command()
def quality(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = _load_config(config)
    run_quality(cfg)

# split: split the dataframe into train, validation and test sets
@app.command()
def split(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = _load_config(config)
    run_split(cfg)

# tune: fine-tune the model on the train and validation sets
@app.command(name="tune")
def fine_tune(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = _load_config(config)
    run_fine_tune(cfg)

# evaluate: evaluate the model on the test set
@app.command()
def evaluate(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = _load_config(config)
    run_evaluate(cfg)


@app.command()
def inference(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = _load_config(config)
    run_inference(cfg)


@app.command()
def explain(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = _load_config(config)
    run_explain(cfg)

# scan: scan the test set with Giskard
@app.command()
def scan(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = _load_config(config)
    run_scan(cfg)

@app.command()
def purge(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    """
    Remove all artifacts (models, splits, previews, metrics) and recreate empty artifacts dir.
    """
    cfg = _load_config(config)
    run_purge(cfg)

# all: run the full pipeline
@app.command()
def all(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    """
    Run the full pipeline.
    """
    cfg = _load_config(config)
    steps = (
        run_load,
        run_clean,
        run_quality,
        run_split,
        run_fine_tune,
        run_evaluate,
        run_scan,
        run_explain,
        run_inference,
    )
    for step in steps:
        step(cfg)
    typer.echo("[all] Pipeline completed.")


if __name__ == "__main__":
    app()


 