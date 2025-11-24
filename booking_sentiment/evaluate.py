from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import DatasetDict
from pathlib import Path
from .config import PathsConfig
from transformers import Trainer, TrainingArguments


def evaluate_model(model_dir: str, tokenized_dir: str, test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate saved HF model on test dataframe and compute metrics.
    """
    print(f"[eval] Evaluating model at '{model_dir}' on {len(test_df)} rows")

    # Load the model and the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    datasets_tokenized = DatasetDict.load_from_disk(tokenized_dir)
    # Load the training arguments
    training_args = TrainingArguments(output_dir=model_dir, do_train=False, do_eval=True, per_device_eval_batch_size=32)
    # Create the trainer
    trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer, eval_dataset=datasets_tokenized["test"])

    # Predict the test set
    pred_output = trainer.predict(datasets_tokenized["test"])
    # Get the logits and the predictions
    logits = torch.from_numpy(pred_output.predictions)
    y_pred_proba = softmax(logits, dim=1)[:, 1].numpy()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    y_test = datasets_tokenized["test"]["label"]
    # Compute metrics: Precision, Recall, F1, AUROC, MCC
    # Precision: the proportion of true positives among all predicted positives
    # Recall: the proportion of true positives among all actual positives
    # F1: the harmonic mean of Precision and Recall
    # AUROC: the area under the ROC curve
    # MCC: the Matthews Correlation Coefficient
    # MCC is a measure of the quality of the classification model
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


