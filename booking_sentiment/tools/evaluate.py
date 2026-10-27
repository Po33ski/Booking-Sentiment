from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

from .inference import SentimentClassifier


def evaluate_model(model_dir: Path, test_df: pd.DataFrame, device: str = "cpu") -> Dict[str, Any]:
    """
    Evaluate the saved HF model on the test dataframe (read from the current test split) and compute metrics.
    """
    print(f"[eval] Evaluating model at '{model_dir}' on {len(test_df)} rows")

    classifier = SentimentClassifier(model_dir, device=device)
    y_pred_proba = classifier.predict_proba(test_df["text"].tolist())
    y_pred = (y_pred_proba >= 0.5).astype(int)
    y_test = test_df["label"].to_numpy()

    # Precision: the proportion of true positives among all predicted positives
    # Recall: the proportion of true positives among all actual positives
    # F1: the harmonic mean of Precision and Recall
    # AUROC: the area under the ROC curve
    # MCC: the Matthews Correlation Coefficient
    metrics = {
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        "AUROC": float(roc_auc_score(y_test, y_pred_proba)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
    }
    print("[eval] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    return metrics
