from pathlib import Path

import pytest
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast

VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "the", "room", "was", "great", "bad", "small", "nice", "staff", "hotel", "dirty", "clean",
]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    """A tiny, randomly initialised DistilBERT classifier saved like the fine-tuned model."""
    model_dir = tmp_path_factory.mktemp("tiny_model")
    vocab_file = model_dir / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab_file))

    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=len(VOCAB), dim=16, n_layers=1, n_heads=2, hidden_dim=32,
        max_position_embeddings=64, num_labels=2,
    )
    model = DistilBertForSequenceClassification(config)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return model_dir
