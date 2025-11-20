from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from cleanlab import Datalab
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from .config import QualityConfig

# get_initial_model_data: get the embeddings and pred_probs from the initial model
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

# Verification: check if the embeddings and pred_probs have the same length and the same dimension
def verify_sample_size(embeddings: np.ndarray, pred_probs: np.ndarray) -> None:
    # Verification: check if the embeddings and pred_probs have the same length and the same dimension
    if len(embeddings) != len(pred_probs):
        raise ValueError(f"Embeddings and pred_probs have different lengths: {len(embeddings)} != {len(pred_probs)}")
    if pred_probs.ndim != 2:
        raise ValueError(f"Pred_probs have wrong dimension: {pred_probs.ndim} != 2")
    assert len(embeddings) == len(pred_probs)
    assert pred_probs.ndim == 2

    print(f"✅ Embeddings shape: {embeddings.shape}")
    print(f"✅ Pred_probs shape: {pred_probs.shape}")
    print(f"✅ Verification passed!")
    return None


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
    verify_sample_size(embeddings, pred_probs)  # Verification: check if the embeddings and pred_probs have the same length and the same dimension

    #build the data dictionary for the CleanLab
    data_dict = {"texts": np.array(texts, dtype=object), "labels": labels}
    lab = Datalab(data_dict, label_name="labels", task="classification")
    # find the issues with the CleanLab
    lab.find_issues(pred_probs=pred_probs, features=embeddings)
    lab.report()

    # near-duplicate issues (report now; drop later after label fixes)
    duplicate_issues = lab.get_issues("near_duplicate")
    duplicate_issues = duplicate_issues[duplicate_issues["is_near_duplicate_issue"]]
    duplicate_issues = duplicate_issues.sort_values(by="near_duplicate_score")
    print(f"[quality] Near-duplicate issues flagged: {len(duplicate_issues)}")

    
    # get the label issues
    label_issues = lab.get_issues("label")
    label_issues = label_issues[label_issues["is_label_issue"]]
    # if there are no label issues, return the dataframe
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

    # Drop rows flagged as near-duplicates (keep representative samples)
    if len(duplicate_issues) > 0:
        to_drop = [i for i in duplicate_issues.index if i in df_fixed.index]
        before = len(df_fixed)
        df_fixed = df_fixed.drop(index=to_drop).reset_index(drop=True)
        print(f"[quality] Dropped {len(to_drop)} near-duplicate rows ({before} -> {len(df_fixed)})")
    else:
        print("[quality] No near-duplicates to drop.")

    # Outlier issues: report and keep (as in the notebook)
    outlier_issues = lab.get_issues("outlier")
    outlier_issues = outlier_issues[outlier_issues["is_outlier_issue"]]
    print(f"[quality] Outliers flagged by CleanLab: {len(outlier_issues)} (kept)")

    return df_fixed


