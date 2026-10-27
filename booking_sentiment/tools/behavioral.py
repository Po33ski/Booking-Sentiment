from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .inference import SentimentClassifier


def build_prediction_function(classifier: SentimentClassifier) -> Callable[[pd.DataFrame], np.ndarray]:
    """
    Giskard prediction function: takes a dataframe with a "text" column and returns
    an (n, 2) array of class probabilities [P(negative), P(positive)].
    """

    def prediction_function(df: pd.DataFrame) -> np.ndarray:
        prob_pos = classifier.predict_proba(df["text"].tolist())
        return np.column_stack([1.0 - prob_pos, prob_pos])

    return prediction_function


def run_giskard_scan(model_dir: Path, test_df: pd.DataFrame, device: str = "cuda", out_dir: Optional[Path] = None):
    """
    Run a Giskard behavioral scan of the saved model (loaded from model_dir) on the test set.
    Saves the markdown report if out_dir is provided.
    """
    from giskard import Dataset as GiskardDataset, Model as GiskardModel, scan

    classifier = SentimentClassifier(model_dir, device=device)

    giskard_dataset = GiskardDataset(
        test_df[["text", "label"]],
        target="label",
        column_types={"text": "text"},
    )
    giskard_model = GiskardModel(
        model=build_prediction_function(classifier),
        model_type="classification",
        classification_labels=[0, 1],
        feature_names=["text"],
    )
    results = scan(giskard_model, giskard_dataset, verbose=True)
    results_md = results.to_markdown()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "giskard_scan.md").write_text(str(results_md), encoding="utf-8")

    return results, results_md
