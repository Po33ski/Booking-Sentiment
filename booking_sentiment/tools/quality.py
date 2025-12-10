from __future__ import annotations

from typing import Tuple
import os

import numpy as np
import pandas as pd
import typer
from cleanlab import Datalab
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from datasets import Dataset
from ..config import QualityConfig

# gets rid of irritating warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class QualityInterface:
    """
    Interface for running CleanLab-based data quality analysis.

    It encapsulates:
    - building embeddings and initial model predictions,
    - removing near-duplicates,
    - fixing label issues,
    - dropping outliers,
    while collecting a textual description of the process.
    """

    def __init__(self, cfg: QualityConfig) -> None:
        self.cfg = cfg
        self._logs: list[str] = []

    # --- logging helpers ---
    def _log(self, message: str = "") -> None:
        self._logs.append(message)
        typer.echo(message)

    def _section_logs(self, start_idx: int) -> str:
        """Return only the logs added since start_idx as a single string."""
        return "\n".join(self._logs[start_idx:])

    # ---get_initial_model_data: get the embeddings and pred_probs from the initial model ---
    def _get_initial_model_data(
        self,
        texts: list[str],
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute embeddings and cross-validated prediction probabilities."""
        model = SentenceTransformer(self.cfg.embedding_model_name, device=self.cfg.device)
        embeddings = model.encode(texts, show_progress_bar=True)

        clf = LogisticRegression(
            random_state=0,
            C=self.cfg.regularization_c,
            solver="liblinear",
        )
        pred_probs = cross_val_predict(
            clf,
            embeddings,
            labels,
            cv=self.cfg.cv_folds,
            method="predict_proba",
            n_jobs=1,
        )
        return embeddings, pred_probs

    def _verify_sample_size(self, embeddings: np.ndarray, pred_probs: np.ndarray) -> None:
        """Raise if embeddings/pred_probs are inconsistent; log basic shapes."""
        same_length = len(embeddings) == len(pred_probs)
        correct_dim = pred_probs.ndim == 2

        if same_length and correct_dim:
            self._log(f"[quality] Embeddings shape: {embeddings.shape}")
            self._log(f"[quality] Pred_probs shape: {pred_probs.shape}")
            self._log("[quality] Verification passed!")
            return

        if not same_length:
            self._log(
                f"❌ Embeddings and pred_probs have different lengths: {len(embeddings)} != {len(pred_probs)}"
            )
        if not correct_dim:
            self._log(f"[quality] Pred_probs have wrong dimension: {pred_probs.ndim} (expected 2)")
        raise ValueError("[quality] Sample verification failed.")

    # --- step 1: remove near-duplicates ---
    def _remove_duplicates(
        self,
        df: pd.DataFrame,
        texts: np.ndarray,
        labels: np.ndarray,
        lab: Datalab,
    ) -> Tuple[pd.DataFrame, str]:
        start_idx = len(self._logs)

        duplicate_issues = lab.get_issues("near_duplicate")
        duplicate_issues = duplicate_issues[duplicate_issues["is_near_duplicate_issue"]]
        duplicate_issues = duplicate_issues.sort_values(by="near_duplicate_score")

        df_deduplicated = df.copy()
        if len(duplicate_issues) == 0:
            self._log("[quality] No near-duplicate issues detected by CleanLab.")
            self._log()
        else:
            self._log(f"[quality] Near-duplicate issues flagged: {len(duplicate_issues)}")
            self._log("all near-duplicate issues:")
            for idx, row in duplicate_issues.iterrows():
                text = texts[idx]
                score = row["near_duplicate_score"]
                label = labels[idx]
                label_name = "POSITIVE" if label == 1 else "NEGATIVE"

                self._log(f"\n--- Near-duplicate (Score: {score:.4f}) ---")
                self._log(f"Label: {label_name} ({label})")
                self._log(f"Text: \"{text}\"")
                self._log("-" * 60)
                self._log()

            df_deduplicated = df_deduplicated.drop(index=duplicate_issues.index)
            df_deduplicated = df_deduplicated.reset_index(drop=True)
            self._log(f"[quality] Deduplicated dataframe: {len(df_deduplicated)}")
            self._log(
                f"[quality] Dropped {len(duplicate_issues)} duplicate rows ({len(df)} -> {len(df_deduplicated)})"
            )
            self._log()

        description = self._section_logs(start_idx)
        return df_deduplicated, description

    # --- step 2: fix label issues ---
    def _fix_labels(
        self,
        df: pd.DataFrame,
        texts: np.ndarray,
        labels: np.ndarray,
        lab: Datalab,
    ) -> Tuple[pd.DataFrame, str]:
        start_idx = len(self._logs)

        label_issues = lab.get_issues("label")
        label_issues = label_issues[label_issues["is_label_issue"]]
        label_issues = label_issues.sort_values(by="label_score")

        df_label_fixed = df.copy()
        if len(label_issues) == 0:
            self._log("[quality] No label issues detected by CleanLab.")
            self._log()
        else:
            self._log(f"[quality] Label issues flagged: {len(label_issues)}")
            top_label_issues_y_true = label_issues.head(10)["given_label"]
            top_label_issues_y_pred = label_issues.head(10)["predicted_label"]
            top_label_issues_idxs = label_issues.head(10).index
            top_label_issues_texts = texts[top_label_issues_idxs]

            self._log("Top 10 label issues")
            for text, y_true, y_pred in zip(
                top_label_issues_texts, top_label_issues_y_true, top_label_issues_y_pred
            ):
                self._log(f"y_true {y_true}, y_pred {y_pred}, text: {text}")
                self._log()

            # make sure we don't get key errors - we removed some rows earlier during deduplication
            label_issues = label_issues[label_issues.index.isin(df.index)]

            idxs = label_issues.index.tolist()
            pred_labels = label_issues["predicted_label"]
            df_label_fixed.loc[idxs, "label"] = pred_labels
            self._log(f"[quality] Fixed labels for {len(idxs)} rows")
            self._log()

        description = self._section_logs(start_idx)
        return df_label_fixed, description

    # --- step 3: drop outlier issues ---
    def _remove_outliers(
        self,
        df: pd.DataFrame,
        texts: np.ndarray,
        labels: np.ndarray,
        lab: Datalab,
    ) -> Tuple[pd.DataFrame, str]:
        start_idx = len(self._logs)

        outlier_issues = lab.get_issues("outlier")
        outlier_issues = outlier_issues[outlier_issues["is_outlier_issue"]]
        outlier_issues = outlier_issues.sort_values(by="outlier_score")

        df_outlier_and_label_fixed = df.copy()
        if len(outlier_issues) == 0:
            self._log("[quality] No outlier issues detected by CleanLab.")
            self._log()
        else:
            self._log(f"[quality] Outliers flagged by CleanLab: {len(outlier_issues)} (kept)")
            self._log("Outliers with the strongest confidence (lowest score):")
            for idx, row in outlier_issues.iterrows():
                text = texts[idx]
                score = row["outlier_score"]
                label = labels[idx]
                label_name = "POSITIVE" if label == 1 else "NEGATIVE"

                self._log(f"\n--- Outlier (Score: {score:.4f}) ---")
                self._log(f"Label: {label_name} ({label})")
                self._log(f"Text: \"{text}\"")
                self._log("-" * 60)
                self._log()

            outlier_issues = outlier_issues[outlier_issues.index.isin(df.index)]
            idxs = outlier_issues.index.tolist()
            df_outlier_and_label_fixed = df_outlier_and_label_fixed.drop(index=idxs)
            self._log(
                f"[quality] Dropped {len(outlier_issues)} outlier rows ({len(df)} -> {len(df_outlier_and_label_fixed)})"
            )
            self._log()

        description = self._section_logs(start_idx)
        return df_outlier_and_label_fixed, description

    # --- public orchestrator: run all steps, optionally multiple iterations ---
    def run(self, df: pd.DataFrame, iterations: int = 1) -> Tuple[pd.DataFrame, str]:
        """
        Run the full CleanLab quality pipeline:
        - compute embeddings and initial probabilities,
        - remove near-duplicates,
        - fix label issues,
        - drop outliers.
        Optionally repeats the whole pipeline `iterations` times, feeding the
        output of one iteration as input to the next.

        Returns the cleaned dataframe and a markdown-friendly description for
        all iterations.
        """
        df_current = df.copy()
        all_descriptions: list[str] = []

        for i in range(iterations):
            self._logs = []

            self._log(f"## Iteration {i+1}")
            self._log("[quality] Running CleanLab data quality")

            texts = df_current["text"].values
            labels = df_current["label"].values

            embeddings, pred_probs = self._get_initial_model_data(
                texts=texts,
                labels=labels,
            )

            self._verify_sample_size(embeddings, pred_probs)

            data_df = df_current[["text", "label"]].copy()
            dataset = Dataset.from_pandas(data_df, preserve_index=False)
            dataset.set_format(type="python")
            lab = Datalab(dataset, label_name="label", task="classification")

            lab.find_issues(pred_probs=pred_probs, features=embeddings)
            lab.report()

            df_deduplicated, _ = self._remove_duplicates(df_current, texts, labels, lab)
            df_label_fixed, _ = self._fix_labels(df_deduplicated, texts, labels, lab)
            df_final, _ = self._remove_outliers(df_label_fixed, texts, labels, lab)

            all_descriptions.append("\n".join(self._logs))
            df_current = df_final

        full_description = "\n\n".join(all_descriptions)
        return df_current, full_description


def run_cleanlab(
    df: pd.DataFrame, cfg: QualityConfig
) -> Tuple[pd.DataFrame, str]:
    """
    Backwards-compatible function that uses the QualityInterface class
    to run the full CleanLab pipeline.
    """
    interface = QualityInterface(cfg)
    return interface.run(df, iterations=cfg.iterations)
