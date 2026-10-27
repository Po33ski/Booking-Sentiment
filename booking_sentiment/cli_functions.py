from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import typer

from .config import ProjectConfig

SPLITS = ("train", "valid", "test")


# ensure_dirs: ensure the artifacts directory exists
def ensure_dirs(cfg: ProjectConfig) -> None:
    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)


def _raw_splits_dir(cfg: ProjectConfig) -> Path:
    """Splits written by `split` (before the quality step)."""
    return cfg.paths.artifacts_dir / "splits_raw"


def _splits_dir(cfg: ProjectConfig) -> Path:
    """Final splits written by `quality` and used for training/evaluation."""
    return cfg.paths.artifacts_dir / "splits"


def _require(path: Path, step: str, hint: str) -> None:
    if not path.exists():
        typer.echo(f"[{step}] '{path}' not found. Run 'booking-sentiment {hint}' first.")
        raise typer.Exit(code=1)


# load: load the raw dataset from HuggingFace and save negative/positive texts
def run_load(cfg: ProjectConfig) -> None:
    from .tools.data_ingest import load_raw_dataset

    ensure_dirs(cfg)
    df_neg, df_pos, df_raw = load_raw_dataset(cfg.dataset)
    out_raw = cfg.paths.artifacts_dir / "raw_preview.parquet"
    out_neg = cfg.paths.artifacts_dir / "neg_preview.parquet"
    out_pos = cfg.paths.artifacts_dir / "pos_preview.parquet"
    df_raw.to_parquet(out_raw, index=False)
    df_neg.to_frame(name="text").to_parquet(out_neg, index=False)
    df_pos.to_frame(name="text").to_parquet(out_pos, index=False)
    typer.echo(f"[load] Preview saved to: {out_raw}, {out_neg}, {out_pos}")


# clean: clean the raw texts and save a parquet file with columns: text, label
def run_clean(cfg: ProjectConfig) -> None:
    from .tools.data_clean import clean_and_label

    ensure_dirs(cfg)
    in_neg = cfg.paths.artifacts_dir / "neg_preview.parquet"
    in_pos = cfg.paths.artifacts_dir / "pos_preview.parquet"
    _require(in_neg, "clean", "load")
    _require(in_pos, "clean", "load")
    df_neg = pd.read_parquet(in_neg)["text"]
    df_pos = pd.read_parquet(in_pos)["text"]
    df_clean = clean_and_label(df_neg, df_pos, cfg.cleaning)
    out_clean = cfg.paths.artifacts_dir / "clean.parquet"
    df_clean.to_parquet(out_clean, index=False)
    typer.echo(f"[clean] Cleaned data saved to: {out_clean}")


# split: sample the cleaned data and split it into train, validation and test sets
def run_split(cfg: ProjectConfig) -> None:
    from .tools.splits import sample_dataframe, split_dataframe

    ensure_dirs(cfg)
    in_path = cfg.paths.artifacts_dir / "clean.parquet"
    _require(in_path, "split", "clean")
    df_clean = pd.read_parquet(in_path)
    df_sampled = sample_dataframe(
        df_clean,
        cfg.dataset.sample_size,
        random_state=cfg.dataset.random_state,
        stratify=cfg.split.stratify,
    )
    parts = split_dataframe(df_sampled, cfg.split)
    # Final splits from a previous run no longer match the new split - remove them,
    # so 'tune' cannot train on stale data before 'quality' is run again.
    stale_dir = _splits_dir(cfg)
    if stale_dir.exists():
        shutil.rmtree(stale_dir)
        typer.echo(f"[split] Removed outdated final splits: {stale_dir} (run 'quality' next)")
    out_dir = _raw_splits_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in zip(SPLITS, parts):
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
    typer.echo(f"[split] Saved splits to: {out_dir}")


# quality: run Cleanlab on train+valid (the test split is kept unchanged)
def run_quality(cfg: ProjectConfig) -> None:
    from .tools.quality import run_cleanlab

    ensure_dirs(cfg)
    raw_dir = _raw_splits_dir(cfg)
    for name in SPLITS:
        _require(raw_dir / f"{name}.parquet", "quality", "split")
    train_df, valid_df, test_df = (pd.read_parquet(raw_dir / f"{name}.parquet") for name in SPLITS)

    df_tv = pd.concat(
        [train_df.assign(split="train"), valid_df.assign(split="valid")],
        ignore_index=True,
    )
    df_fixed, description = run_cleanlab(df_tv, test_df, cfg.quality)
    typer.echo(f"[quality] Completed {cfg.quality.iterations} quality iteration(s).")

    out_dir = _splits_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "valid"):
        part = df_fixed[df_fixed["split"] == name].drop(columns="split").reset_index(drop=True)
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
        typer.echo(f"[quality] {name}: {len(part)} rows")
    test_df.to_parquet(out_dir / "test.parquet", index=False)
    typer.echo(f"[quality] test: {len(test_df)} rows (unchanged)")
    typer.echo(f"[quality] Final splits saved to: {out_dir}")

    # Save the quality report as markdown in a separate folder.
    quality_logs_dir = cfg.paths.artifacts_dir / "quality_logs"
    quality_logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = quality_logs_dir / "quality_log.md"
    log_path.write_text("# Cleanlab data quality report\n\n" + description, encoding="utf-8")
    typer.echo(f"[quality] Quality log saved to: {log_path}")


# tune: fine-tune the model on the train and validation sets
def run_fine_tune(cfg: ProjectConfig) -> None:
    from .tools.fine_tune import fine_tune

    ensure_dirs(cfg)
    splits_dir = _splits_dir(cfg)
    _require(splits_dir / "train.parquet", "tune", "quality")
    _require(splits_dir / "valid.parquet", "tune", "quality")
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    valid_df = pd.read_parquet(splits_dir / "valid.parquet")
    model_dir, _ = fine_tune(
        train_df, valid_df, cfg.fine_tune, cfg.paths, lowercase=cfg.cleaning.lowercase_text
    )
    typer.echo(f"[tune] Model artifact at: {model_dir}")


# evaluate: evaluate the model on the test set
def run_evaluate(cfg: ProjectConfig) -> None:
    from .tools.evaluate import evaluate_model

    ensure_dirs(cfg)
    test_path = _splits_dir(cfg) / "test.parquet"
    model_dir = cfg.paths.artifacts_dir / "finetuned_model"
    _require(test_path, "evaluate", "quality")
    _require(model_dir, "evaluate", "tune")
    test_df = pd.read_parquet(test_path)
    metrics = evaluate_model(model_dir, test_df, device=cfg.fine_tune.device)
    out = cfg.paths.artifacts_dir / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(f"[evaluate] Metrics saved to: {out}")


# explain: explain the model predictions on a few test samples
def run_explain(cfg: ProjectConfig) -> None:
    from .tools.explain import explain_samples

    ensure_dirs(cfg)
    test_path = _splits_dir(cfg) / "test.parquet"
    model_dir = cfg.paths.artifacts_dir / "finetuned_model"
    _require(test_path, "explain", "quality")
    _require(model_dir, "explain", "tune")
    test_df = pd.read_parquet(test_path)
    out = explain_samples(model_dir, test_df, cfg.paths.artifacts_dir / "explain", visualize=True)
    typer.echo(f"[explain] Saved attribution TSV/HTML files to: {out}")


# scan: scan the model on the test set with Giskard
def run_scan(cfg: ProjectConfig) -> None:
    from .tools.behavioral import run_giskard_scan

    ensure_dirs(cfg)
    test_path = _splits_dir(cfg) / "test.parquet"
    model_dir = cfg.paths.artifacts_dir / "finetuned_model"
    _require(test_path, "scan", "quality")
    _require(model_dir, "scan", "tune")
    test_df = pd.read_parquet(test_path)
    out = cfg.paths.artifacts_dir / "giskard"
    _, results_md = run_giskard_scan(model_dir, test_df, device=cfg.fine_tune.device, out_dir=out)
    typer.echo("[scan] Giskard scan completed. Results:")
    typer.echo(f"[scan] {results_md}")
    typer.echo(f"[scan] Giskard scan saved to: {out}")


# inference: prompt for reviews and classify them with the fine-tuned model
def run_inference(cfg: ProjectConfig) -> None:
    """
    Interactive inference loop that keeps accepting inputs until the user quits.
    """
    from .tools.inference import load_sentiment_classifier

    model_dir = cfg.paths.artifacts_dir / "finetuned_model"
    _require(model_dir, "inference", "tune")

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


def _is_protected_dir(path: Path) -> bool:
    """True for directories that must never be wiped (project root, home, filesystem root, parents of cwd)."""
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    return (
        resolved in {cwd, Path.home().resolve(), Path(resolved.anchor)}
        or resolved in cwd.parents
        or (resolved / ".git").exists()
        or (resolved / "pyproject.toml").exists()
    )


# purge: delete the artifacts directory and recreate it empty
def run_purge(cfg: ProjectConfig, assume_yes: bool = False) -> None:
    artifacts_dir = cfg.paths.artifacts_dir
    if _is_protected_dir(artifacts_dir):
        typer.echo(f"[purge] Refusing to delete '{artifacts_dir.resolve()}' - it is not an artifacts directory.")
        raise typer.Exit(code=1)
    if artifacts_dir.exists():
        if not assume_yes and not typer.confirm(f"Delete everything in '{artifacts_dir.resolve()}'?"):
            typer.echo("[purge] Aborted.")
            return
        shutil.rmtree(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"[purge] Cleaned artifacts directory: {artifacts_dir}")
