from __future__ import annotations

import re

import pandas as pd
import typer

from ..config import CleaningConfig


def remove_terms_regex(terms: list[str]) -> str:
    """
    Build a pattern matching texts that are exactly one of `terms`.
    Terms are escaped, so characters like "." or "(" are matched literally.
    Case-insensitivity is handled by Pandas via case=False.
    """
    return "|".join(f"^{re.escape(term)}$" for term in terms)


def _filter_texts(texts: pd.Series, label: int, pattern: str, cfg: CleaningConfig) -> pd.DataFrame:
    texts = texts.fillna("").astype(str).str.strip()
    # An empty pattern would match every text, so only filter when there are terms to remove.
    if pattern:
        texts = texts[~texts.str.contains(pattern, case=False, regex=True)]
    texts = texts[texts.str.len() >= cfg.min_text_len]
    return pd.DataFrame({"text": texts.to_numpy(), "label": label})


def clean_and_label(df_neg: pd.Series, df_pos: pd.Series, cfg: CleaningConfig) -> pd.DataFrame:
    """
    Apply simple cleaning rules and return a unified dataframe with columns: text, label.
    label=0 -> negative, label=1 -> positive
    """
    typer.echo("[clean] Cleaning texts and building labeled dataframe")
    pattern = remove_terms_regex(cfg.remove_terms)
    df_neg_f = _filter_texts(df_neg, 0, pattern, cfg)
    df_pos_f = _filter_texts(df_pos, 1, pattern, cfg)

    typer.echo(f"[clean] Negative samples after filtering: {len(df_neg_f)}")
    typer.echo(f"[clean] Positive samples after filtering: {len(df_pos_f)}")

    df_all = pd.concat([df_neg_f, df_pos_f], ignore_index=True)
    if cfg.lowercase_text:
        df_all["text"] = df_all["text"].str.lower()

    # Deduplicate (optionally ignoring case).
    key = df_all["text"].str.casefold() if cfg.casefold_dedup else df_all["text"]
    df_all = df_all.assign(_key=key).drop_duplicates(subset=["_key", "label"])

    # The same text cannot be both positive and negative - drop such conflicting texts.
    conflicting = df_all.groupby("_key")["label"].transform("nunique") > 1
    typer.echo(f"[clean] Dropped {int(conflicting.sum())} rows whose text appears with both labels")
    df_all = df_all[~conflicting].drop(columns="_key").reset_index(drop=True)

    typer.echo(f"[clean] Final rows: {len(df_all)}")
    return df_all
