from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .config import ProjectConfig
from .data_ingest import load_raw_dataset
from .data_clean import clean_and_label
from .quality import run_cleanlab_stub
from .splits import split_dataframe
from .train_hf import train_stub
from .evaluate import evaluate_stub
from .behavior import scan_stub
from .explain import explain_stub

app = typer.Typer(add_completion=False, help="Booking Sentiment - MLOps-friendly CLI")


def ensure_dirs(cfg: ProjectConfig) -> None:
    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)


@app.command()
def load(config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to JSON config file")) -> None:
    cfg = ProjectConfig.load(config)
    ensure_dirs(cfg)
    neg, pos = load_raw_dataset(cfg.dataset)
    # Save quick snapshot for debugging
    out = cfg.paths.artifacts_dir / "raw_preview.parquet"
    df = pd.DataFrame({"neg": neg.head(5), "pos": pos.head(5)})
    df.to_parquet(out, index=False)
    typer.echo(f"[load] Preview saved to: {out}")


@app.command()
def clean(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    ensure_dirs(cfg)
    neg, pos = load_raw_dataset(cfg.dataset)
    df = clean_and_label(neg, pos, cfg.cleaning)
    out = cfg.paths.artifacts_dir / "clean.parquet"
    df.to_parquet(out, index=False)
    typer.echo(f"[clean] Cleaned data saved to: {out}")


@app.command()
def quality(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    cfg = ProjectConfig.load(config)
    ensure_dirs(cfg)
    in_path = cfg.paths.artifacts_dir / "clean.parquet"
    if not in_path.exists():
        typer.echo("[quality] 'clean.parquet' not found. Run 'uv run clean' first.")
        raise typer.Exit(code=1)
    df = pd.read_parquet(in_path)
    df2 = run_cleanlab_stub(df, cfg.quality)
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
    model_dir, _ = train_stub(train_df, valid_df, cfg.train, cfg.paths)
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
    metrics = evaluate_stub(str(model_dir), test_df)
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
    _ = ProjectConfig.load(config)
    explain_stub()


@app.command()
def all(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    """
    Run the full lightweight pipeline with stubs.
    """
    cfg = ProjectConfig.load(config)
    ensure_dirs(cfg)
    # load + clean
    neg, pos = load_raw_dataset(cfg.dataset)
    df = clean_and_label(neg, pos, cfg.cleaning)
    (cfg.paths.artifacts_dir / "splits").mkdir(parents=True, exist_ok=True)
    # quality (stub)
    df = run_cleanlab_stub(df, cfg.quality)
    # split
    if cfg.dataset.sample_size:
        df = df.sample(n=cfg.dataset.sample_size, random_state=cfg.dataset.random_state, replace=False)
        df = df.reset_index(drop=True)
    train_df, valid_df, test_df = split_dataframe(df, cfg.split)
    train_df.to_parquet(cfg.paths.artifacts_dir / "splits/train.parquet", index=False)
    valid_df.to_parquet(cfg.paths.artifacts_dir / "splits/valid.parquet", index=False)
    test_df.to_parquet(cfg.paths.artifacts_dir / "splits/test.parquet", index=False)
    # train (stub)
    model_dir, _ = train_stub(train_df, valid_df, cfg.train, cfg.paths)
    # evaluate (stub)
    _ = evaluate_stub(str(model_dir), test_df)
    typer.echo("[all] Pipeline completed (with stubs).")


if __name__ == "__main__":
    app()


