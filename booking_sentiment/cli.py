from __future__ import annotations

from typing import Optional
import typer

from .config import ProjectConfig
from .cli_functions import run_load, run_clean, run_quality, run_split, run_train, run_evaluate, run_explain, run_scan, run_purge
# app: root of the CLI
app = typer.Typer(add_completion=False, help="Booking Sentiment - MLOps-friendly CLI")


# load: load the raw dataset from HuggingFace and return positive/negative Series
@app.command()
def load(config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to JSON config file")) -> None:
    cfg = ProjectConfig.load(config)
    run_load(cfg)

# clean: clean the raw dataset and return a dataframe with columns: text, label
@app.command()
def clean(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    run_clean(cfg)

# quality: run the quality analysis (CleanLab) and return a dataframe with columns: text, label
@app.command()
def quality(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    run_quality(cfg)

# split: split the dataframe into train, validation and test sets
@app.command()
def split(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    run_split(cfg)

# train: train the model on the train and validation sets
@app.command(name="train")
def train_cmd(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    run_train(cfg)

# evaluate: evaluate the model on the test set
@app.command()
def evaluate(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    run_evaluate(cfg)


@app.command()
def explain(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    run_explain(cfg)

# scan: scan the test set with Giskard
@app.command()
def scan(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    run_scan(cfg)

@app.command()
def purge(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    """
    Remove all artifacts (models, splits, previews, metrics) and recreate empty artifacts dir.
    """
    cfg = ProjectConfig.load(config)
    run_purge(cfg)

# all: run the full pipeline
@app.command()
def all(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    """
    Run the full pipeline.
    """
    cfg = ProjectConfig.load(config)
    steps = (
        run_load,
        run_clean,
        run_quality,
        run_split,
        run_train,
        run_evaluate,
    )
    for step in steps:
        step(cfg)
    typer.echo("[all] Pipeline completed.")


if __name__ == "__main__":
    app()


 