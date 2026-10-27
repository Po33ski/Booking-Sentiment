from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import typer
from cleanlab import Datalab
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors

from ..config import QualityConfig
from ..runtime import resolve_device

# gets rid of irritating warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LABEL_NAMES = {0: "NEGATIVE", 1: "POSITIVE"}


@dataclass
class QualityFixes:
    """
    Changes chosen by Cleanlab for one iteration.
    All values are row positions (0..n-1) in the dataframe that was analysed.
    """

    label_fixes: dict[int, int] = field(default_factory=dict)  # position -> new label
    duplicates_to_drop: list[int] = field(default_factory=list)
    outliers_to_drop: list[int] = field(default_factory=list)


def select_duplicates_to_drop(
    near_duplicate_sets: Sequence[Sequence[int]],
    flagged: Sequence[int],
    labels: np.ndarray,
) -> list[int]:
    """
    Keep one example from every near-duplicate group and drop the others.
    If the members of a group have different labels, the whole group is dropped,
    because there is no way to tell which label is right.
    """
    dropped: set[int] = set()
    for idx in sorted(int(i) for i in flagged):
        if idx in dropped:
            continue
        group = ({idx} | {int(j) for j in near_duplicate_sets[idx]}) - dropped
        if len({int(labels[i]) for i in group}) > 1:
            dropped |= group
        else:
            dropped |= group - {idx}
    return sorted(dropped)


def apply_fixes(df: pd.DataFrame, fixes: QualityFixes) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply label fixes and row drops by position. Returns the fixed dataframe
    (with a fresh 0..n-1 index) and the positions of the rows that were kept,
    so callers can keep other arrays (e.g. embeddings) aligned.
    """
    out = df.reset_index(drop=True).copy()
    if fixes.label_fixes:
        positions = list(fixes.label_fixes.keys())
        out.loc[positions, "label"] = list(fixes.label_fixes.values())
    to_drop = set(fixes.duplicates_to_drop) | set(fixes.outliers_to_drop)
    kept = np.array([i for i in range(len(out)) if i not in to_drop], dtype=int)
    return out.iloc[kept].reset_index(drop=True), kept


def find_test_leaks(
    embeddings: np.ndarray, test_embeddings: np.ndarray, min_similarity: float
) -> np.ndarray:
    """
    Return positions of rows whose cosine similarity to the closest test row is >= min_similarity.
    """
    if len(embeddings) == 0 or len(test_embeddings) == 0:
        return np.array([], dtype=int)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(test_embeddings)
    distances, _ = nn.kneighbors(embeddings)
    similarity = 1.0 - distances[:, 0]
    return np.flatnonzero(similarity >= min_similarity)


class QualityInterface:
    """
    Cleanlab-based data quality step for the train+valid data.

    The test split is never modified: it is only used to remove train/valid rows
    that (almost) duplicate a test row, so the evaluation is not leaked.
    For train+valid it:
    - fixes label issues,
    - keeps one example per near-duplicate group,
    - drops outliers,
    while collecting a textual description of the process.
    """

    def __init__(self, cfg: QualityConfig) -> None:
        self.cfg = cfg
        self._logs: list[str] = []
        self._encoder = None

    # --- logging helpers ---
    def _log(self, message: str = "", echo: bool = True) -> None:
        """Add a line to the report; `echo=False` keeps per-row details out of the console."""
        self._logs.append(message)
        if echo:
            typer.echo(message)

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                self.cfg.embedding_model_name, device=resolve_device(self.cfg.device)
            )
        return self._encoder.encode(
            list(texts), show_progress_bar=True, normalize_embeddings=True
        )

    def _cross_val_pred_probs(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Out-of-fold class probabilities from a logistic regression on the embeddings."""
        clf = LogisticRegression(
            random_state=0,
            C=self.cfg.regularization_c,
            solver="liblinear",
        )
        return cross_val_predict(
            clf,
            embeddings,
            labels,
            cv=self.cfg.cv_folds,
            method="predict_proba",
            n_jobs=1,
        )

    def _verify_sample_size(self, embeddings: np.ndarray, pred_probs: np.ndarray) -> None:
        """Raise if embeddings/pred_probs are inconsistent; log basic shapes."""
        if len(embeddings) != len(pred_probs) or pred_probs.ndim != 2:
            raise ValueError(
                f"[quality] Sample verification failed: embeddings {embeddings.shape}, "
                f"pred_probs {pred_probs.shape}"
            )
        self._log(f"[quality] Embeddings shape: {embeddings.shape}")
        self._log(f"[quality] Pred_probs shape: {pred_probs.shape}")

    def _log_rows(self, title: str, positions: Sequence[int], texts: np.ndarray, labels: np.ndarray) -> None:
        for pos in positions:
            label = int(labels[pos])
            self._log(f"\n--- {title} ---", echo=False)
            self._log(f"Label: {LABEL_NAMES.get(label, label)} ({label})", echo=False)
            self._log(f'Text: "{texts[pos]}"', echo=False)
        if len(positions):
            self._log(echo=False)

    # --- one Cleanlab pass: decide what to change (by position) ---
    def _find_fixes(self, df: pd.DataFrame, embeddings: np.ndarray) -> QualityFixes:
        texts = df["text"].to_numpy()
        labels = df["label"].to_numpy()

        pred_probs = self._cross_val_pred_probs(embeddings, labels)
        self._verify_sample_size(embeddings, pred_probs)

        lab = Datalab(df[["text", "label"]], label_name="label", task="classification")
        lab.find_issues(
            pred_probs=pred_probs,
            features=embeddings,
            issue_types={"label": {}, "near_duplicate": {}, "outlier": {}},
        )

        fixes = QualityFixes()

        # step 1: label issues -> take the label predicted by Cleanlab
        label_issues = lab.get_issues("label")
        label_issues = label_issues[label_issues["is_label_issue"]].sort_values(by="label_score")
        fixes.label_fixes = {
            int(pos): int(new)
            for pos, new in label_issues["predicted_label"].items()
            if int(new) != int(labels[pos])
        }
        self._log(f"[quality] Label issues flagged and fixed: {len(fixes.label_fixes)}")
        for pos, new in list(fixes.label_fixes.items())[:10]:
            self._log(f"y_true {labels[pos]}, y_fixed {new}, text: {texts[pos]}", echo=False)
        fixed_labels = labels.copy()
        for pos, new in fixes.label_fixes.items():
            fixed_labels[pos] = new

        # step 2: near-duplicates -> keep one example per group (using the fixed labels)
        dup_issues = lab.get_issues("near_duplicate")
        flagged = np.flatnonzero(dup_issues["is_near_duplicate_issue"].to_numpy())
        near_duplicate_sets = lab.get_info("near_duplicate")["near_duplicate_sets"]
        fixes.duplicates_to_drop = select_duplicates_to_drop(near_duplicate_sets, flagged, fixed_labels)
        self._log(
            f"[quality] Near-duplicates flagged: {len(flagged)}, "
            f"dropped: {len(fixes.duplicates_to_drop)} (one example per group is kept)"
        )
        self._log_rows("Dropped near-duplicate", fixes.duplicates_to_drop, texts, fixed_labels)

        # step 3: outliers -> drop
        outlier_issues = lab.get_issues("outlier")
        outlier_issues = outlier_issues[outlier_issues["is_outlier_issue"]].sort_values(by="outlier_score")
        fixes.outliers_to_drop = [int(p) for p in outlier_issues.index]
        self._log(f"[quality] Outliers flagged and dropped: {len(fixes.outliers_to_drop)}")
        self._log_rows("Dropped outlier", fixes.outliers_to_drop, texts, fixed_labels)

        return fixes

    # --- public orchestrator ---
    def run(self, df: pd.DataFrame, test_df: pd.DataFrame, iterations: int = 1) -> Tuple[pd.DataFrame, str]:
        """
        Clean `df` (train+valid) and return it together with a markdown report.
        Extra columns in `df` (e.g. "split") are preserved.
        """
        self._logs = []
        df_current = df.reset_index(drop=True).copy()

        self._log("## Test leakage check")
        embeddings = self._embed(df_current["text"].tolist())
        test_embeddings = self._embed(test_df["text"].tolist())
        leaks = find_test_leaks(embeddings, test_embeddings, self.cfg.test_leak_similarity)
        self._log(
            f"[quality] Train/valid rows with cosine similarity >= {self.cfg.test_leak_similarity} "
            f"to a test row: {len(leaks)} (dropped)"
        )
        self._log_rows(
            "Dropped (too similar to a test row)", leaks, df_current["text"].to_numpy(), df_current["label"].to_numpy()
        )
        df_current, kept = apply_fixes(df_current, QualityFixes(duplicates_to_drop=leaks.tolist()))
        embeddings = embeddings[kept]

        for i in range(iterations):
            self._log(f"\n## Iteration {i + 1}")
            self._log("[quality] Running Cleanlab data quality")
            fixes = self._find_fixes(df_current, embeddings)
            n_before = len(df_current)
            df_current, kept = apply_fixes(df_current, fixes)
            # Reuse the embeddings of the kept rows instead of encoding them again.
            embeddings = embeddings[kept]
            self._log(f"[quality] Rows: {n_before} -> {len(df_current)}")

        return df_current, "\n".join(self._logs)


def run_cleanlab(df: pd.DataFrame, test_df: pd.DataFrame, cfg: QualityConfig) -> Tuple[pd.DataFrame, str]:
    """
    Convenience function used by the CLI.
    """
    interface = QualityInterface(cfg)
    return interface.run(df, test_df, iterations=cfg.iterations)
