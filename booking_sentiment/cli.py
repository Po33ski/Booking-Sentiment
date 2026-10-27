from __future__ import annotations

from typing import Optional

import typer

from .cli_functions import (
    run_clean,
    run_evaluate,
    run_explain,
    run_fine_tune,
    run_inference,
    run_load,
    run_purge,
    run_quality,
    run_scan,
    run_split,
)
from .config import ProjectConfig
from .runtime import configure_runtime

# app: root of the CLI
app = typer.Typer(add_completion=False, help="Booking Sentiment - MLOps-friendly CLI")

ConfigOption = typer.Option(None, "--config", "-c", help="Path to JSON config file")


# Shared helper to hydrate config and apply deterministic settings
def _load_config(config_path: Optional[str]) -> ProjectConfig:
    try:
        cfg = ProjectConfig.load(config_path)
    except (FileNotFoundError, ValueError) as e:  # pydantic's ValidationError is a ValueError
        typer.echo(f"[config] Invalid configuration: {e}", err=True)
        raise typer.Exit(code=2)
    configure_runtime(cfg.fine_tune.seed, cfg.fine_tune.device)
    return cfg


@app.command()
def load(config: Optional[str] = ConfigOption) -> None:
    """Download the raw dataset from Hugging Face and save negative/positive texts."""
    run_load(_load_config(config))


@app.command()
def clean(config: Optional[str] = ConfigOption) -> None:
    """Clean the texts and build a labeled dataframe (text, label)."""
    run_clean(_load_config(config))


@app.command()
def split(config: Optional[str] = ConfigOption) -> None:
    """Sample the cleaned data and split it into train/valid/test."""
    run_split(_load_config(config))


@app.command()
def quality(config: Optional[str] = ConfigOption) -> None:
    """Run Cleanlab on train+valid (test stays unchanged)."""
    run_quality(_load_config(config))


@app.command(name="tune")
def fine_tune(config: Optional[str] = ConfigOption) -> None:
    """Fine-tune the model on the train split (model selection on valid)."""
    run_fine_tune(_load_config(config))


@app.command()
def evaluate(config: Optional[str] = ConfigOption) -> None:
    """Evaluate the fine-tuned model on the test split."""
    run_evaluate(_load_config(config))


@app.command()
def scan(config: Optional[str] = ConfigOption) -> None:
    """Run a Giskard behavioral scan on the test split."""
    run_scan(_load_config(config))


@app.command()
def explain(config: Optional[str] = ConfigOption) -> None:
    """Save Input x Gradient token attributions for a few test samples."""
    run_explain(_load_config(config))


@app.command()
def inference(config: Optional[str] = ConfigOption) -> None:
    """Interactively classify reviews typed in the terminal."""
    run_inference(_load_config(config))


@app.command()
def purge(
    config: Optional[str] = ConfigOption,
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation"),
) -> None:
    """Remove all artifacts (models, splits, previews, metrics) and recreate an empty artifacts dir."""
    run_purge(_load_config(config), assume_yes=yes)


@app.command(name="all")
def run_all(config: Optional[str] = ConfigOption) -> None:
    """
    Run the full (non-interactive) pipeline. Use 'inference' afterwards to try the model.
    """
    cfg = _load_config(config)
    steps = (
        run_load,
        run_clean,
        run_split,
        run_quality,
        run_fine_tune,
        run_evaluate,
        run_scan,
        run_explain,
    )
    for step in steps:
        step(cfg)
    typer.echo("[all] Pipeline completed. Run 'booking-sentiment inference' to try the model.")


if __name__ == "__main__":
    app()
