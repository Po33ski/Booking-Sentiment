from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from giskard import Model as GiskardModel, Dataset as GiskardDataset, scan
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

def run_giskard_scan(model_dir: Path, test_df: pd.DataFrame, device: str = "cuda", out_dir: Optional[Path] = None):
    """
    Run Giskard behavioral scan using a saved model (loaded from model_dir).
    Mirrors the approach from the notebook: build a pipeline and wrap a prediction_function.
    Saves a simple text summary if out_dir is provided.
    """
    device_resolved = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"


    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model.eval()
    # Create the model pipeline: this is the pipeline that will be used to predict the sentiment of the text
    model_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device_resolved,
    )
    
    # Prediction function: takes a dataframe and returns the predictions
    @torch.no_grad()
    def prediction_function(df: pd.DataFrame) -> np.ndarray:
        preds = model_pipeline(df["text"].tolist())
        # As in the notebook, use the 'score' returned by the pipeline.
        y_pred_proba = [float(p["score"]) for p in preds]
        return np.array(y_pred_proba, dtype=float)

    # Create the Giskard dataset: this is the dataset that will be used to scan the test set.
    # The dataset is created using the test set and the target column.
    giskard_dataset = GiskardDataset(test_df, target="label")
    # Create the Giskard model. This is the model that will be used to scan the test set. 
    # The model is created using the prediction function and the test set.
    giskard_model = GiskardModel(
        model=prediction_function,
        model_type="classification",
        classification_labels=[0, 1],
        feature_names=["text"],
    )
    # Scan the test set using the Giskard model
    results = scan(giskard_model, giskard_dataset, verbose=False)
    print(results)
    results_md = results.to_markdown()
    # Save the results to a file
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save a minimal textual summary
        summary_path = Path(out_dir) / "giskard_scan.txt"
        try:
            summary_path.write_text(str(results_md), encoding="utf-8")
        except Exception:
            pass

    return results, results_md


