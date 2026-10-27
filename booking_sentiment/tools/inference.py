from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..runtime import resolve_device

__all__ = ["MAX_LENGTH", "SentimentClassifier", "load_sentiment_classifier"]

# Maximum number of tokens per text; longer texts are truncated.
MAX_LENGTH = 512


class SentimentClassifier:
    """
    Lightweight wrapper around the fine-tuned HuggingFace model.
    Label 0 = negative, label 1 = positive.
    """

    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """
        Return P(positive) (probability of label 1) for every text.
        """
        texts = list(texts)
        probs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
            ).to(self.device)
            logits = self.model(**encoded).logits.float()
            probs.append(softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(probs) if probs else np.empty(0, dtype=np.float32)

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Return the label and P(positive) for the provided `text`.
        """
        prob_pos = float(self.predict_proba([text])[0])
        label = "positive" if prob_pos >= 0.5 else "negative"
        return label, prob_pos


def load_sentiment_classifier(model_dir: Path, device: str = "cpu") -> SentimentClassifier:
    """
    Convenience factory used by the CLI.
    """
    return SentimentClassifier(model_dir, device=device)
