from __future__ import annotations

import pandas as pd

from .config import QualityConfig


def run_cleanlab_stub(df: pd.DataFrame, cfg: QualityConfig) -> pd.DataFrame:
    """
    Placeholder for CleanLab-driven data quality step.
    Currently returns df unchanged and prints a message.
    """
    print(
        f"[quality] (stub) Would compute embeddings '{cfg.embedding_model_name}', "
        f"CV folds={cfg.cv_folds}, logistic C={cfg.logistic_c}"
    )
    return df


