from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import SplitConfig


def sample_dataframe(
    df: pd.DataFrame, sample_size: Optional[int], random_state: int, stratify: bool = True
) -> pd.DataFrame:
    """
    Draw `sample_size` rows (stratified by label). If `sample_size` is None or not
    smaller than the dataframe, all rows are returned.
    """
    if sample_size is None or sample_size >= len(df):
        print(f"[split] Using all {len(df)} rows (sample_size={sample_size})")
        return df.reset_index(drop=True)
    sampled, _ = train_test_split(
        df,
        train_size=sample_size,
        random_state=random_state,
        stratify=df["label"] if stratify else None,
    )
    print(f"[split] Sampled {len(sampled)} of {len(df)} rows")
    return sampled.reset_index(drop=True)


def split_dataframe(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deterministic split into train/valid/test according to fractions in cfg.
    """
    stratify_labels = df["label"] if cfg.stratify else None

    train_valid, test = train_test_split(
        df,
        test_size=cfg.test_frac,  # fraction of the data used for testing
        random_state=cfg.random_state,
        stratify=stratify_labels,
    )
    # recompute valid frac relative to the remaining data
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
