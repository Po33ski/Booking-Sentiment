from __future__ import annotations

import re
from typing import Tuple

import pandas as pd

from .config import CleaningConfig


def compile_remove_regex(terms: list[str]) -> re.Pattern[str]:
    escaped = [re.escape(t) for t in terms]
    patterns = [f"^(?:{e})$" for e in escaped]
    return re.compile("|".join(patterns), flags=re.IGNORECASE)


def clean_and_label(neg: pd.Series, pos: pd.Series, cfg: CleaningConfig) -> pd.DataFrame:
    """
    Apply simple cleaning rules and return a unified dataframe with columns: text, label.
    label=0 -> negative, label=1 -> positive
    """
    print("[clean] Cleaning texts and building labeled dataframe")
    remove_re = compile_remove_regex(cfg.remove_terms)

    neg_f = neg[~neg.str.contains(remove_re, na=True)]
    pos_f = pos[~pos.str.contains(remove_re, na=True)]

    neg_f = neg_f[neg_f.str.len() >= cfg.min_text_len]
    pos_f = pos_f[pos_f.str.len() >= cfg.min_text_len]

    df_neg = neg_f.reset_index(drop=True).to_frame(name="text")
    df_pos = pos_f.reset_index(drop=True).to_frame(name="text")
    df_neg["label"] = 0
    df_pos["label"] = 1

    df = pd.concat([df_neg, df_pos], ignore_index=True)
    df = df.drop_duplicates(ignore_index=True)

    if cfg.casefold_dedup:
        df["_lower"] = df["text"].str.lower()
        before = len(df)
        df = df.drop_duplicates(subset="_lower")
        df = df.drop(columns="_lower")
        print(f"[clean] Case-insensitive dedup: {before} -> {len(df)}")

    print(f"[clean] Final rows: {len(df)}")
    return df


