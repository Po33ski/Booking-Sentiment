from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

__all__ = ["SentimentClassifier", "load_sentiment_classifier"]


class SentimentClassifier:
    """
    Lightweight wrapper around the fine-tuned HuggingFace model for local inference.
    """

    def __init__(self, model_dir: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Return the label and P(positive) for the provided `text`.
        """
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits
            prob_pos = softmax(logits, dim=1)[0, 1].item()
        label = "positive" if prob_pos >= 0.5 else "negative"
        return label, prob_pos


def load_sentiment_classifier(model_dir: Path) -> SentimentClassifier:
    """
    Convenience factory used by the CLI.
    """
    return SentimentClassifier(model_dir)

