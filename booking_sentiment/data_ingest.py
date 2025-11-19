from __future__ import annotations

from typing import Tuple

import pandas as pd
from datasets import load_dataset

from .config import DatasetConfig


def load_raw_dataset(cfg: DatasetConfig) -> Tuple[pd.Series, pd.Series]:
    """
    Load booking reviews dataset from HuggingFace and return positive/negative Series.
    """
    print(f"[ingest] Loading dataset '{cfg.dataset_name}' split='{cfg.split}'")
    ds = load_dataset(cfg.dataset_name)[cfg.split].to_pandas()
    neg = ds[cfg.negative_col].astype(str).str.strip()
    pos = ds[cfg.positive_col].astype(str).str.strip()
    print(f"[ingest] Loaded rows: {len(ds)}")
    return neg, pos


