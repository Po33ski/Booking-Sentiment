from __future__ import annotations

from typing import Tuple

import pandas as pd
from datasets import load_dataset

from ..config import DatasetConfig


# load_raw_dataset: load the raw dataset from HuggingFace and return negative/positive Series
def load_raw_dataset(cfg: DatasetConfig) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Load booking reviews dataset from HuggingFace and return negative/positive Series.
    """
    print(f"[ingest] Loading dataset '{cfg.dataset_name}'")
    df_raw = load_dataset(cfg.dataset_name)["train"].to_pandas()
    # Missing values become empty strings (not the text "nan"), so the clean step drops them.
    df_neg = df_raw[cfg.negative_col].fillna("").astype(str).str.strip()
    df_pos = df_raw[cfg.positive_col].fillna("").astype(str).str.strip()
    print(f"[ingest] Loaded rows: {len(df_raw)}")
    print("Negative reviews:", len(df_neg))
    print("Positive reviews:", len(df_pos))
    return df_neg, df_pos, df_raw
