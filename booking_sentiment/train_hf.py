from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from .config import TrainConfig, PathsConfig


def train_stub(train_df: pd.DataFrame, valid_df: pd.DataFrame, train_cfg: TrainConfig, paths: PathsConfig) -> Tuple[Path, Dict[str, Any]]:
    """
    Placeholder for HF fine-tuning training step.
    Returns a dummy model path and metrics dict.
    """
    out_dir = paths.artifacts_dir / "finetuned_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[train] (stub) Would fine-tune '{train_cfg.model_name}' for {train_cfg.epochs} epochs "
        f"lr={train_cfg.learning_rate} use_cpu={train_cfg.use_cpu}"
    )
    # dummy model file
    dummy_model = out_dir / "DUMMY_MODEL.txt"
    dummy_model.write_text("Replace with actual model files.", encoding="utf-8")
    metrics = {"MCC": None, "F1": None}
    return out_dir, metrics


