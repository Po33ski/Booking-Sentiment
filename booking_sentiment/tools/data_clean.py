from __future__ import annotations

import re
import typer
from typing import Tuple

import pandas as pd

from ..config import CleaningConfig


def remove_terms_regex(terms: list[str]) -> str:
    to_remove = [f"^{word}$" for word in terms]
    to_remove = "|".join(to_remove)
    # Return string pattern; case-insensitivity handled by Pandas via case=False
    return to_remove


def clean_and_label(df_neg: pd.DataFrame, df_pos: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """
    Apply simple cleaning rules and return a unified dataframe with columns: text, label.
    label=0 -> negative, label=1 -> positive
    """
    typer.echo("[clean] Cleaning texts and building labeled dataframe")
    remove_re = remove_terms_regex(cfg.remove_terms)
    # remove the terms from the negative and positive reviews
    neg_f = df_neg[~df_neg.str.contains(remove_re, case=False, na=True, regex=True)]
    pos_f = df_pos[~df_pos.str.contains(remove_re, case=False, na=True, regex=True)]

    # remove the empty strings
    neg_f = neg_f[neg_f.str.len() >= cfg.min_text_len]
    pos_f = pos_f[pos_f.str.len() >= cfg.min_text_len]

    # build the dataframe with columns: text, label
    df_neg_f = neg_f.reset_index(drop=True).to_frame().rename(columns={"Negative_Review": "text"})
    df_pos_f = pos_f.reset_index(drop=True).to_frame().rename(columns={"Positive_Review": "text"})
    # label the negative reviews as 0 and the positive reviews as 1
    df_neg_f["label"] = 0
    df_pos_f["label"] = 1

    typer.echo(f"[clean] Negative samples after filtering: {len(df_neg_f)}")
    typer.echo(f"[clean] Positive samples after filtering: {len(df_pos_f)}")

    # concatenate the negative and positive reviews and drop duplicates
    df_all = pd.concat([df_neg_f, df_pos_f], ignore_index=True)
    # lowercase the text
    if cfg.lowercase_text:
        df_all["text"] = df_all["text"].str.lower()
    # drop duplicates   
    df_all = df_all.drop_duplicates(ignore_index=True)




    typer.echo(f"[clean] Final rows: {len(df_all)}")
    return df_all


