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
    # remove the terms from the negative and positive reviews
    neg_f = neg[~neg.str.contains(remove_re, case=False, na=True, regex=True)]
    pos_f = pos[~pos.str.contains(remove_re, case=False, na=True, regex=True)]

    # remove the empty strings
    neg_f = neg_f[neg_f.str.len() >= cfg.min_text_len]
    pos_f = pos_f[pos_f.str.len() >= cfg.min_text_len]

    # build the dataframe with columns: text, label
    df_neg = neg_f.reset_index(drop=True).to_frame(name="text")
    df_pos = pos_f.reset_index(drop=True).to_frame(name="text")
    # label the negative reviews as 0 and the positive reviews as 1
    df_neg["label"] = 0
    df_pos["label"] = 1

    print("Negative samples after filtering:", len(df_neg))
    print("Positive samples after filtering:", len(df_pos))

    # concatenate the negative and positive reviews and drop duplicates
    df = pd.concat([df_neg, df_pos], ignore_index=True)
    # lowercase the text
    if cfg.lowercase_text:
        df["text"] = df["text"].str.lower()
    # drop duplicates   
    df = df.drop_duplicates(ignore_index=True)

    # # case-insensitive deduplication
    # if cfg.casefold_dedup:
    #     df["_lower"] = df["text"].str.lower()
    #     before = len(df)
    #     df = df.drop_duplicates(subset="_lower")
    #     df = df.drop(columns="_lower")
    #     print(f"[clean] Case-insensitive dedup: {before} -> {len(df)}")


    print(f"[clean] Final rows: {len(df)}")
    return df


