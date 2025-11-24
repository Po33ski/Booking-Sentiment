from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from cleanlab import Datalab
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from datasets import Dataset
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
    # 1. Compute text embeddings with sentence-transformers
    model = SentenceTransformer(embedding_model_name, device=device)
    embeddings = model.encode(texts, show_progress_bar=True)
    # 2. Create LogisticRegression object
    clf = LogisticRegression(random_state=0, C=regularization_c, solver="liblinear")
    # 3. Train logistic regression and get probability predictions using cross_val_predict
    pred_probs = cross_val_predict(
        clf,
        embeddings,
        labels,
        cv=cv_folds,
        method="predict_proba",
        n_jobs=1,
    )
    return embeddings, pred_probs 

# Verification: check if the embeddings and pred_probs have the same length and the same dimension
def verify_sample_size(embeddings: np.ndarray, pred_probs: np.ndarray) -> bool:
    # Verification: check if the embeddings and pred_probs have the same length and the same dimension
    same_length = len(embeddings) == len(pred_probs)
    correct_dim = pred_probs.ndim == 2

    if same_length and correct_dim:
        print(f"✅ Embeddings shape: {embeddings.shape}")
        print(f"✅ Pred_probs shape: {pred_probs.shape}")
        print("✅ Verification passed!")
        return True

    if not same_length:
        print(
            f"❌ Embeddings and pred_probs have different lengths: {len(embeddings)} != {len(pred_probs)}"
        )
    if not correct_dim:
        print(f"❌ Pred_probs have wrong dimension: {pred_probs.ndim} (expected 2)")

    return False


# run_cleanlab: run the quality analysis (CleanLab) and return a dataframe with columns: text, label
def run_cleanlab(df: pd.DataFrame, cfg: QualityConfig) -> pd.DataFrame:
    """
    Detect issues with CleanLab and fix label issues by replacing labels with predicted labels
    for rows flagged as is_label_issue. Returns modified dataframe.
    """
    print("[quality] Running CleanLab data quality")
    texts = df["text"].values
    labels = df["label"].values

    embeddings, pred_probs = _get_initial_model_data(
        texts=texts,
        labels=labels,
        embedding_model_name=cfg.embedding_model_name,
        cv_folds=cfg.cv_folds,
        regularization_c=cfg.regularization_c,
        device=cfg.device,
    )
    # verify the sample size and dimension
    if not verify_sample_size(embeddings, pred_probs):
        raise ValueError("[quality] Sample verification failed.")
        
    # create a dataframe with columns: text, label
    data_df = df[['text', 'label']].copy()     # df is this 5k sample
    dataset = Dataset.from_pandas(data_df, preserve_index=False)
    dataset.set_format(type="python")  # or type="numpy"
    lab = Datalab(dataset, label_name="label", task="classification")

    # find the issues with the CleanLab
    lab.find_issues(pred_probs=pred_probs, features=embeddings)
    lab.report()

    # near-duplicate issues (report now; drop later after label fixes)
    duplicate_issues = lab.get_issues("near_duplicate")
    duplicate_issues = duplicate_issues[duplicate_issues["is_near_duplicate_issue"]]
    duplicate_issues = duplicate_issues.sort_values(by="near_duplicate_score")

    df_deduplicated = df.copy()
    # if there are no near-duplicate issues, print a message
    if len(duplicate_issues) == 0:
        print("[quality] No near-duplicate issues detected by CleanLab.")
        print()
    else:
        print(f"[quality] Near-duplicate issues flagged: {len(duplicate_issues)}")
        # Remove the case-sensitive duplicates
        df["text_lower"] = df["text"].str.lower()
        df_deduplicated = df.drop_duplicates(subset="text_lower")
        df_deduplicated = df_deduplicated.drop(columns="text_lower")
        df_deduplicated = df_deduplicated.reset_index(drop=True)
        print(f"[quality] Deduplicated dataframe: {len(df_deduplicated)}")
        print()
    
    # get the label issues
    label_issues = lab.get_issues("label")
    label_issues = label_issues[label_issues["is_label_issue"]]
    label_issues = label_issues.sort_values(by="label_score")

    df_label_fixed = df_deduplicated.copy()
    # if there are no label issues, print a message
    if len(label_issues) == 0:
        print("[quality] No label issues detected by CleanLab.")
        print()
    else:
        print(f"[quality] Label issues flagged: {len(label_issues)}")
        top_label_issues_y_true = label_issues.head(10)["given_label"]
        top_label_issues_y_pred = label_issues.head(10)["predicted_label"]
        top_label_issues_idxs = label_issues.head(10).index
        top_label_issues_texts = texts[top_label_issues_idxs]

        print("Top 10 label issues")
        for text, y_true, y_pred in zip(top_label_issues_texts, top_label_issues_y_true, top_label_issues_y_pred):
            print(f"y_true {y_true}, y_pred {y_pred}, text: {text}")
            print()
        
        # make sure we don't get key errors - we removed some rows earlier during deduplication
        label_issues = label_issues[label_issues.index.isin(df_deduplicated.index)]

        idxs = label_issues.index.tolist()
        pred_labels = label_issues["predicted_label"]
        # fix the labels
        df_label_fixed.loc[idxs, "label"] = pred_labels
        print(f"[quality] Fixed labels for {len(idxs)} rows")
        print()
    
    # Outlier issues: report and keep (as in the notebook)
    outlier_issues = lab.get_issues("outlier")
    outlier_issues = outlier_issues[outlier_issues["is_outlier_issue"]]
    outlier_issues = outlier_issues.sort_values(by="outlier_score")

    df_outlier_and_label_fixed = df_label_fixed.copy()
    if len(outlier_issues) == 0:
        print("[quality] No outlier issues detected by CleanLab.")
        print()
    else:
        print(f"[quality] Outliers flagged by CleanLab: {len(outlier_issues)} (kept)")
        print("Outliers with the strongest confidence (lowest score):")
        for idx, row in outlier_issues.iterrows():
            text = texts[idx]
            score = row["outlier_score"]
            label = labels[idx]
            label_name = "POSITIVE" if label == 1 else "NEGATIVE"
            
            print(f"\n--- Outlier (Score: {score:.4f}) ---")
            print(f"Label: {label_name} ({label})")
            print(f"Text: \"{text}\"")
            print("-" * 60)
            print()

        outlier_issues = outlier_issues[outlier_issues.index.isin(df_label_fixed.index)]
        
        idxs = outlier_issues.index.tolist()
        df_outlier_and_label_fixed = df_outlier_and_label_fixed.drop(index=idxs)
        print(f"[quality] Dropped {len(outlier_issues)} outlier rows ({len(df_outlier_and_label_fixed)} -> {len(df_outlier_and_label_fixed)})")
        print()


    return df_outlier_and_label_fixed


