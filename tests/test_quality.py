import numpy as np
import pandas as pd

from booking_sentiment.config import QualityConfig
from booking_sentiment.tools.quality import (
    QualityFixes,
    QualityInterface,
    apply_fixes,
    find_test_leaks,
    select_duplicates_to_drop,
)


def test_apply_fixes_targets_the_right_rows():
    # Regression test: label fixes and outlier drops used to hit shifted rows
    # after duplicates were dropped and the index was reset.
    df = pd.DataFrame({"text": list("abcdef"), "label": [0, 0, 0, 0, 0, 1]}, index=[10, 11, 12, 13, 14, 15])
    fixes = QualityFixes(label_fixes={4: 1}, duplicates_to_drop=[1], outliers_to_drop=[5])

    out, kept = apply_fixes(df, fixes)

    assert out["text"].tolist() == ["a", "c", "d", "e"]
    assert out["label"].tolist() == [0, 0, 0, 1]
    assert kept.tolist() == [0, 2, 3, 4]


def test_one_example_per_duplicate_group_is_kept():
    sets = [[1], [0], [], [4], [3]]
    labels = np.array([0, 0, 1, 1, 0])
    # group {0, 1} has one label -> keep 0, drop 1; group {3, 4} has conflicting labels -> drop both
    assert select_duplicates_to_drop(sets, flagged=[0, 1, 3, 4], labels=labels) == [1, 3, 4]


def test_find_test_leaks():
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    test_emb = np.array([[1.0, 0.01]])
    assert find_test_leaks(emb, test_emb, 0.99).tolist() == [0]


def test_quality_run_end_to_end(monkeypatch):
    rng = np.random.default_rng(0)
    n = 120
    labels = np.array([0, 1] * (n // 2))
    texts = [f"text {i}" for i in range(n)]
    centers = np.where(labels[:, None] == 1, 1.0, -1.0) * np.ones((n, 8))
    emb = centers + rng.normal(scale=0.3, size=(n, 8))
    emb[1] = emb[0] = emb[0]  # an exact near-duplicate pair
    labels[1] = labels[0]
    df = pd.DataFrame({"text": texts, "label": labels, "split": "train"})
    test_df = pd.DataFrame({"text": ["test 0"], "label": [0]})
    lookup = {t: e for t, e in zip(texts, emb)}
    lookup["test 0"] = np.full(8, 50.0) * np.array([1, -1] * 4)

    def fake_embed(self, items):
        return np.stack([lookup[t] for t in items])

    monkeypatch.setattr(QualityInterface, "_embed", fake_embed)
    out, report = QualityInterface(QualityConfig(cv_folds=3)).run(df, test_df)

    assert "split" in out.columns
    assert not ({"text 0", "text 1"} <= set(out["text"])), "only one copy of the duplicate pair may remain"
    assert set(out["text"]) <= set(texts)
    assert "Iteration 1" in report
