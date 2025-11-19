from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SplitConfig


def split_dataframe(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deterministic split into train/valid/test according to fractions in cfg.
    """
    cfg.validate_sum()
    stratify_labels = df["label"] if cfg.stratify else None

    test_size = cfg.test_frac
    train_valid, test = train_test_split(
        df,
        test_size=test_size,
        random_state=cfg.random_state,
        stratify=stratify_labels,
    )
    # recompute valid frac relative to remaining
    valid_rel = cfg.valid_frac / (cfg.train_frac + cfg.valid_frac)
    stratify_labels_tv = train_valid["label"] if cfg.stratify else None
    train, valid = train_test_split(
        train_valid,
        test_size=valid_rel,
        random_state=cfg.random_state,
        stratify=stratify_labels_tv,
    )
    print(f"[split] train={len(train)} valid={len(valid)} test={len(test)}")
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


