import numpy as np
import pandas as pd
import torch

from booking_sentiment.tools.behavioral import build_prediction_function
from booking_sentiment.tools.evaluate import evaluate_model
from booking_sentiment.tools.explain import explain_samples
from booking_sentiment.tools.inference import SentimentClassifier

TEXTS = ["the room was great", "the room was dirty", "nice staff", "bad hotel"]


def test_predict_proba_returns_positive_class_probability(tiny_model_dir):
    clf = SentimentClassifier(tiny_model_dir)
    probs = clf.predict_proba(TEXTS, batch_size=3)

    expected = []
    for text in TEXTS:
        enc = clf.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            expected.append(torch.softmax(clf.model(**enc).logits, dim=-1)[0, 1].item())
    np.testing.assert_allclose(probs, expected, atol=1e-5)


def test_giskard_prediction_function_returns_both_classes(tiny_model_dir):
    # Regression test: Giskard used to get the top-class score, which is always >= 0.5.
    clf = SentimentClassifier(tiny_model_dir)
    out = build_prediction_function(clf)(pd.DataFrame({"text": TEXTS}))
    assert out.shape == (len(TEXTS), 2)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(out[:, 1], clf.predict_proba(TEXTS), atol=1e-6)


def test_explain_writes_files(tiny_model_dir, tmp_path):
    # Regression test: explain used to return without computing anything.
    test_df = pd.DataFrame({"text": TEXTS, "label": [1, 0, 1, 0]})
    out = explain_samples(tiny_model_dir, test_df, tmp_path / "explain", num_pos=2, num_neg=2, visualize=True)

    tsvs = sorted(out.glob("sample_*.tsv"))
    assert len(tsvs) == 4 and len(list(out.glob("sample_*.html"))) == 4
    table = pd.read_csv(tsvs[0], sep="\t", comment="#")
    assert table["token"].tolist() == ["the", "room", "was", "great"]  # no [CLS]/[SEP]


def test_evaluate_model_metrics(tiny_model_dir):
    metrics = evaluate_model(tiny_model_dir, pd.DataFrame({"text": TEXTS, "label": [1, 0, 1, 0]}))
    assert set(metrics) == {"Precision", "Recall", "F1", "AUROC", "MCC"}
