from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def evaluate_stub(model_dir: str, test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Placeholder for evaluation step.
    """
    print(f"[eval] (stub) Would evaluate model at '{model_dir}' on {len(test_df)} rows")
    return {"Precision": None, "Recall": None, "F1": None, "AUROC": None, "MCC": None}


