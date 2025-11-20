from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def evaluate_model(model_dir: str, test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate saved HF model on test dataframe and compute metrics.
    """
    print(f"[eval] Evaluating model at '{model_dir}' on {len(test_df)} rows")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    texts = test_df["text"].astype(str).tolist()
    y_test = test_df["label"].to_numpy()

    all_logits = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        all_logits.append(logits)

    logits = torch.cat(all_logits, dim=0)
    y_pred_proba = softmax(logits, dim=1)[:, 1].numpy()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "Precision": float(precision_score(y_test, y_pred)),
        "Recall": float(recall_score(y_test, y_pred)),
        "F1": float(f1_score(y_test, y_pred)),
        "AUROC": float(roc_auc_score(y_test, y_pred_proba)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
    }
    print(
        "[eval] "
        + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    )
    return metrics


