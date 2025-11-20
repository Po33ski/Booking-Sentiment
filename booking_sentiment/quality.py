from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from cleanlab import Datalab
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from .config import QualityConfig


def _get_initial_model_data(
    texts: list[str],
    labels: np.ndarray,
    embedding_model_name: str,
    cv_folds: int,
    regularization_c: float,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    model = SentenceTransformer(embedding_model_name, device=device)
    embeddings = model.encode(texts, show_progress_bar=True)
    clf = LogisticRegression(random_state=0, C=regularization_c, solver="liblinear")
    pred_probs = cross_val_predict(
        clf,
        embeddings,
        labels,
        cv=cv_folds,
        method="predict_proba",
        n_jobs=-1,
    )
    return embeddings, pred_probs

# run_cleanlab: run the quality analysis (CleanLab) and return a dataframe with columns: text, label
def run_cleanlab(df: pd.DataFrame, cfg: QualityConfig) -> pd.DataFrame:
    """
    Detect issues with CleanLab and fix label issues by replacing labels with predicted labels
    for rows flagged as is_label_issue. Returns modified dataframe.
    """
    print("[quality] Running CleanLab data quality")
    texts = df["text"].astype(str).tolist()
    labels = df["label"].to_numpy()

    embeddings, pred_probs = _get_initial_model_data(
        texts=texts,
        labels=labels,
        embedding_model_name=cfg.embedding_model_name,
        cv_folds=cfg.cv_folds,
        regularization_c=cfg.regularization_c,
    
        device=cfg.device,
    )

    data_dict = {"texts": np.array(texts, dtype=object), "labels": labels}
    lab = Datalab(data_dict, label_name="labels", task="classification")
    lab.find_issues(pred_probs=pred_probs, features=embeddings)

    label_issues = lab.get_issues("label")
    label_issues = label_issues[label_issues["is_label_issue"]]
    if len(label_issues) == 0:
        print("[quality] No label issues detected by CleanLab.")
        return df

    idxs = label_issues.index.tolist()
    pred_labels = label_issues["predicted_label"]
    df_fixed = df.copy()
    mask = [i in df_fixed.index for i in idxs]
    fixed_indices = [i for i, m in zip(idxs, mask) if m]
    df_fixed.loc[fixed_indices, "label"] = pred_labels.loc[fixed_indices]
    print(f"[quality] Fixed labels for {len(fixed_indices)} rows")
    return df_fixed


