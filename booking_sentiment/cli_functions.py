from __future__ import annotations

import shutil
from sklearn.utils import resample
import pandas as pd
import typer

from .config import ProjectConfig
from .tools.data_ingest import load_raw_dataset
from .tools.data_clean import clean_and_label
from .tools.quality import run_cleanlab
from .tools.splits import split_dataframe
from .tools.train import train
from .tools.evaluate import evaluate_model
from .tools.behavioral import run_giskard_scan
from .tools.explain import explain_samples
from .tools.inference import load_sentiment_classifier






# ensure_dirs: ensure the directories exist for the artifacts (splits, models, etc.)
def ensure_dirs(cfg: ProjectConfig) -> None:
    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

# load: load the raw dataset from HuggingFace and return positive/negative Series
def run_load(cfg: ProjectConfig) -> None:
    ensure_dirs(cfg)
    df_neg, df_pos, df_raw = load_raw_dataset(cfg.dataset)
    out_raw = cfg.paths.artifacts_dir / "raw_preview.parquet"
    out_neg = cfg.paths.artifacts_dir / "neg_preview.parquet"
    out_pos = cfg.paths.artifacts_dir / "pos_preview.parquet"
    df_raw.to_parquet(out_raw, index=False)
    df_neg.to_frame(name="text").to_parquet(out_neg, index=False)
    df_pos.to_frame(name="text").to_parquet(out_pos, index=False)
    typer.echo(f"[load] Preview saved to: {out_raw}, {out_neg}, {out_pos}")


# clean: clean the raw dataset and return a dataframe with columns: text, label
def run_clean(cfg: ProjectConfig) -> None:
    ensure_dirs(cfg)
    in_neg = cfg.paths.artifacts_dir / "neg_preview.parquet"
    in_pos = cfg.paths.artifacts_dir / "pos_preview.parquet"
    if not in_neg.exists() or not in_pos.exists():
        typer.echo("[clean] 'neg_preview.parquet' or 'pos_preview.parquet' not found. Run 'uv run load' first.")
        raise typer.Exit(code=1)
    df_neg = pd.read_parquet(in_neg)["text"]
    df_pos = pd.read_parquet(in_pos)["text"]
    df_clean = clean_and_label(df_neg, df_pos, cfg.cleaning)
    out_clean = cfg.paths.artifacts_dir / "clean.parquet"
    df_clean.to_parquet(out_clean, index=False)
    typer.echo(f"[clean] Cleaned data saved to: {out_clean}")


# quality: run the quality analysis (CleanLab) and return a dataframe with columns: text, label
def run_quality(cfg: ProjectConfig) -> None:
    ensure_dirs(cfg)
    in_path = cfg.paths.artifacts_dir / "clean.parquet"
    if not in_path.exists():
        typer.echo("[quality] 'clean.parquet' not found. Run 'uv run clean' first.")
        raise typer.Exit(code=1)
    df_clean = pd.read_parquet(in_path)
    if cfg.dataset.sample_size:
        assert cfg.dataset.sample_size > 0, "Sample size must be greater than 0"
        df_clean = resample(
            df_clean,
            replace=False,
            n_samples=cfg.dataset.sample_size,
            random_state=cfg.dataset.random_state,
            stratify=df_clean["label"],
        )
        df_clean = df_clean.reset_index(drop=True)
    for i in range(cfg.quality.iterations):
        df_fixed = run_cleanlab(df_clean, cfg.quality)
        typer.echo(f"[quality] Iteration {i+1} completed")
    out = cfg.paths.artifacts_dir / "quality_fixed.parquet"
    df_fixed.to_parquet(out, index=False)
    typer.echo(f"[quality] Quality output saved to: {out}")


# split: split the dataframe into train, validation and test sets
def run_split(cfg: ProjectConfig) -> None:
    ensure_dirs(cfg)
    in_path = cfg.paths.artifacts_dir / "quality_fixed.parquet"
    if not in_path.exists():
        typer.echo("[split] No input data. Run 'uv run clean' (and optionally 'uv run quality') first.")
        raise typer.Exit(code=1)
    df_fixed = pd.read_parquet(in_path)
    if cfg.dataset.sample_size and len(df_fixed) <= cfg.dataset.sample_size:
        # sample the dataframe to the sample size
        sample_size_fixed = min(len(df_fixed), cfg.dataset.sample_size)
        if sample_size_fixed <= len(df_fixed):
            df_fixed = df_fixed.sample(n=sample_size_fixed, random_state=cfg.dataset.random_state, replace=False)
            df_fixed = df_fixed.reset_index(drop=True)
            typer.echo(f"[split] Subsampled to {sample_size_fixed} rows")
    train_df, valid_df, test_df = split_dataframe(df_fixed, cfg.split)
    (cfg.paths.artifacts_dir / "splits").mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(cfg.paths.artifacts_dir / "splits/train.parquet", index=False)
    valid_df.to_parquet(cfg.paths.artifacts_dir / "splits/valid.parquet", index=False)
    test_df.to_parquet(cfg.paths.artifacts_dir / "splits/test.parquet", index=False)
    typer.echo("[split] Saved splits to artifacts/splits")


# train: train the model on the train and validation sets
def run_train(cfg: ProjectConfig) -> None:
    ensure_dirs(cfg)
    splits_dir = cfg.paths.artifacts_dir / "splits"
    if not (splits_dir / "train.parquet").exists():
        typer.echo("[train] Splits not found. Run 'uv run split' first.")
        raise typer.Exit(code=1)
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    valid_df = pd.read_parquet(splits_dir / "valid.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    model_dir, _ = train(train_df, valid_df, test_df, cfg.train, cfg.paths)
    typer.echo(f"[train] Model artifact at: {model_dir}")

# evaluate: evaluate the model on the test set
def run_evaluate(cfg: ProjectConfig) -> None:
    ensure_dirs(cfg)
    splits_dir = cfg.paths.artifacts_dir / "splits"
    if not (splits_dir / "test.parquet").exists():
        typer.echo("[evaluate] Test split not found. Run 'uv run split' first.")
        raise typer.Exit(code=1)
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    model_dir = cfg.paths.artifacts_dir / "finetuned_model"
    tokenized_dir = cfg.paths.artifacts_dir / "tokenized"
    metrics = evaluate_model(model_dir, tokenized_dir, len(test_df))
    out = cfg.paths.artifacts_dir / "metrics.json"
    import json as _json

    out.write_text(_json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(f"[evaluate] Metrics saved to: {out}")


# explain: explain the model on the test set
def run_explain(cfg: ProjectConfig) -> None:
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


# scan: scan the test set with Giskard
def run_scan(cfg: ProjectConfig) -> None:
    ensure_dirs(cfg)
    test_path = cfg.paths.artifacts_dir / "splits/test.parquet"
    if not test_path.exists():
        typer.echo("[scan] Test split not found. Run 'uv run split' first.")
        raise typer.Exit(code=1)
    test_df = pd.read_parquet(test_path)
    model_dir = cfg.paths.artifacts_dir / "finetuned_model"
    out = cfg.paths.artifacts_dir / "giskard"
    run_giskard_scan(model_dir, test_df, device=getattr(cfg.quality, "device", "cpu"), out_dir=out)
    typer.echo(f"[scan] Giskard scan saved to: {out}")

# inference: prompt for reviews and classify them with the fine-tuned model
def run_inference(cfg: ProjectConfig) -> None:
    """
    Interactive inference loop that keeps accepting inputs until the user quits.
    """
    ensure_dirs(cfg)
    model_dir = cfg.paths.artifacts_dir / "finetuned_model"
    if not model_dir.exists():
        typer.echo("[inference] 'finetuned_model' not found. Run 'uv run train' first.")
        raise typer.Exit(code=1)

    classifier = load_sentiment_classifier(model_dir)
    typer.echo("[inference] Type a review to classify (q/quit to exit).")

    while True:
        text = typer.prompt("Enter a comment").strip()
        if text.lower() in {"q", "quit"}:
            typer.echo("[inference] Exiting inference loop.")
            break
        if not text:
            typer.echo("[inference] Empty input. Provide text or type 'quit'.")
            continue

        label, prob_pos = classifier.predict(text)
        typer.echo(f"[inference] Result: {label} (P(pos)={prob_pos:.3f})")

# purge: purge the artifacts directory
def run_purge(cfg: ProjectConfig) -> None:
    artifacts_dir = cfg.paths.artifacts_dir
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"[purge] Cleaned artifacts directory: {artifacts_dir}")