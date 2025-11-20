from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .config import ProjectConfig
from .data_ingest import load_raw_dataset
from .data_clean import clean_and_label
from .quality import run_cleanlab
from .splits import split_dataframe
from .train_hf import train_hf
from .evaluate import evaluate_model
from .behavior import scan_stub
from .explain import explain_samples
from .check_duplicates import check_duplicates
# app: root of the CLI
app = typer.Typer(add_completion=False, help="Booking Sentiment - MLOps-friendly CLI")


# ensure_dirs: ensure the directories exist for the artifacts (splits, models, etc.)
def ensure_dirs(cfg: ProjectConfig) -> None:
    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

# load: load the raw dataset from HuggingFace and return positive/negative Series
@app.command()
def load(config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to JSON config file")) -> None:
    # load the configuration from a JSON file or return the default configuration
    cfg = ProjectConfig.load(config)
    ensure_dirs(cfg)
    neg, pos = load_raw_dataset(cfg.dataset)
    raw_df = pd.DataFrame({"neg": neg, "pos": pos})
    # Save quick snapshot for debugging
    check_duplicates(neg, pos, raw_df)
    out = cfg.paths.artifacts_dir / "raw_preview.parquet"
    # smoke test: save the first 5 rows of the negative and positive reviews
    df = pd.DataFrame({"neg": neg.head(5), "pos": pos.head(5)})
    df.to_parquet(out, index=False)
    typer.echo(f"[load] Preview saved to: {out}")

# clean: clean the raw dataset and return a dataframe with columns: text, label
@app.command()
def clean(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    ensure_dirs(cfg)
    neg, pos = load_raw_dataset(cfg.dataset)
    df = clean_and_label(neg, pos, cfg.cleaning)
    out = cfg.paths.artifacts_dir / "clean.parquet"
    df.to_parquet(out, index=False)
    typer.echo(f"[clean] Cleaned data saved to: {out}")

# quality: run the quality analysis (CleanLab) and return a dataframe with columns: text, label
@app.command()
def quality(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    ensure_dirs(cfg)
    in_path = cfg.paths.artifacts_dir / "clean.parquet"
    if not in_path.exists():
        typer.echo("[quality] 'clean.parquet' not found. Run 'uv run clean' first.")
        raise typer.Exit(code=1)
    df = pd.read_parquet(in_path)
    df2 = run_cleanlab(df, cfg.quality)
    out = cfg.paths.artifacts_dir / "quality.parquet"
    df2.to_parquet(out, index=False)
    typer.echo(f"[quality] Quality output saved to: {out}")


@app.command()
def split(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    ensure_dirs(cfg)
    in_path = cfg.paths.artifacts_dir / "quality.parquet"
    if not in_path.exists():
        # fallback to clean data
        in_path = cfg.paths.artifacts_dir / "clean.parquet"
    if not in_path.exists():
        typer.echo("[split] No input data. Run 'uv run clean' (and optionally 'uv run quality') first.")
        raise typer.Exit(code=1)
    df = pd.read_parquet(in_path)
    if cfg.dataset.sample_size:
        df = df.sample(n=cfg.dataset.sample_size, random_state=cfg.dataset.random_state, replace=False)
        df = df.reset_index(drop=True)
        typer.echo(f"[split] Subsampled to {len(df)} rows")
    train_df, valid_df, test_df = split_dataframe(df, cfg.split)
    (cfg.paths.artifacts_dir / "splits").mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(cfg.paths.artifacts_dir / "splits/train.parquet", index=False)
    valid_df.to_parquet(cfg.paths.artifacts_dir / "splits/valid.parquet", index=False)
    test_df.to_parquet(cfg.paths.artifacts_dir / "splits/test.parquet", index=False)
    typer.echo("[split] Saved splits to artifacts/splits")


@app.command(name="train")
def train_cmd(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    ensure_dirs(cfg)
    splits_dir = cfg.paths.artifacts_dir / "splits"
    if not (splits_dir / "train.parquet").exists():
        typer.echo("[train] Splits not found. Run 'uv run split' first.")
        raise typer.Exit(code=1)
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    valid_df = pd.read_parquet(splits_dir / "valid.parquet")
    model_dir, _ = train_hf(train_df, valid_df, cfg.train, cfg.paths)
    typer.echo(f"[train] Model artifact at: {model_dir}")


@app.command()
def evaluate(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    ensure_dirs(cfg)
    splits_dir = cfg.paths.artifacts_dir / "splits"
    if not (splits_dir / "test.parquet").exists():
        typer.echo("[evaluate] Test split not found. Run 'uv run split' first.")
        raise typer.Exit(code=1)
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    model_dir = cfg.paths.artifacts_dir / "finetuned_model"
    metrics = evaluate_model(str(model_dir), test_df)
    out = cfg.paths.artifacts_dir / "metrics.json"
    import json as _json

    out.write_text(_json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(f"[evaluate] Metrics saved to: {out}")


@app.command()
def scan(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    ensure_dirs(cfg)
    test_path = cfg.paths.artifacts_dir / "splits/test.parquet"
    if not test_path.exists():
        typer.echo("[scan] Test split not found. Run 'uv run split' first.")
        raise typer.Exit(code=1)
    test_df = pd.read_parquet(test_path)
    scan_stub(test_df)


@app.command()
def explain(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    ensure_dirs(cfg)
    splits_dir = cfg.paths.artifacts_dir / "splits"
    if not (splits_dir / "test.parquet").exists():
        typer.echo("[explain] Test split not found. Run 'uv run split' first.")
        raise typer.Exit(code=1)
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    model_dir = cfg.paths.artifacts_dir / "finetuned_model"
    out = cfg.paths.artifacts_dir / "explain"
    explain_samples(model_dir, test_df, out)
    typer.echo(f"[explain] Saved attribution TSVs to: {out}")


@app.command()
def all(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    """
    Run the full pipeline.
    """
    cfg = ProjectConfig.load(config)
    ensure_dirs(cfg)
    # load + clean
    neg, pos = load_raw_dataset(cfg.dataset)
    df = clean_and_label(neg, pos, cfg.cleaning)
    (cfg.paths.artifacts_dir / "splits").mkdir(parents=True, exist_ok=True)
    # quality
    df = run_cleanlab(df, cfg.quality)
    # split
    if cfg.dataset.sample_size:
        df = df.sample(n=cfg.dataset.sample_size, random_state=cfg.dataset.random_state, replace=False)
        df = df.reset_index(drop=True)
    train_df, valid_df, test_df = split_dataframe(df, cfg.split)
    train_df.to_parquet(cfg.paths.artifacts_dir / "splits/train.parquet", index=False)
    valid_df.to_parquet(cfg.paths.artifacts_dir / "splits/valid.parquet", index=False)
    test_df.to_parquet(cfg.paths.artifacts_dir / "splits/test.parquet", index=False)
    # train
    model_dir, _ = train_hf(train_df, valid_df, cfg.train, cfg.paths)
    # evaluate
    _ = evaluate_model(str(model_dir), test_df)
    typer.echo("[all] Pipeline completed.")


if __name__ == "__main__":
    app()


