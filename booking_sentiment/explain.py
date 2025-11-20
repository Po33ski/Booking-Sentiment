from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from captum.attr import InputXGradient, configure_interpretable_embedding_layer
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _compute_attributions(model, tokenizer, text: str) -> Tuple[List[str], List[float], float, int]:
    tokenizer_output = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    tokens = tokenizer.convert_ids_to_tokens(tokenizer_output["input_ids"][0])

    model_copy = deepcopy(model)
    interpretable_embedding_layer = configure_interpretable_embedding_layer(
        model_copy, "distilbert.embeddings"
    )
    input_embeddings = interpretable_embedding_layer.indices_to_embeddings(
        tokenizer_output["input_ids"]
    )

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, inputs, attention_mask):
            return self.m(inputs, attention_mask=attention_mask)[0]

    model_wrapper = Wrapper(model_copy)
    input_x_gradient = InputXGradient(model_wrapper)
    attributions = input_x_gradient.attribute(
        input_embeddings,
        target=1,
        additional_forward_args=tokenizer_output["attention_mask"],
    )

    attributions = attributions.sum(dim=-1).squeeze(0)
    norm = torch.norm(attributions)
    attributions = attributions / (norm + 1e-12)
    attributions = [float(a) for a in attributions]

    with torch.no_grad():
        logits = model(**tokenizer_output).logits
        y_pred_proba = softmax(logits, dim=1)[0, 1].item()
        y_pred = int(y_pred_proba >= 0.5)

    # remove special tokens
    tokens = tokens[1:-1]
    attributions = attributions[1:-1]
    return tokens, attributions, y_pred_proba, y_pred


def explain_samples(model_dir: Path, test_df: pd.DataFrame, out_dir: Path, num_pos: int = 3, num_neg: int = 3) -> Path:
    """
    Compute InputXGradient attributions for first N positive and N negative samples and save TSVs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    pos_samples = test_df[test_df["label"] == 1].head(num_pos)
    neg_samples = test_df[test_df["label"] == 0].head(num_neg)

    outputs = []
    for row in pd.concat([pos_samples, neg_samples]).itertuples(index=False):
        text, label = row.text, int(row.label)
        tokens, attrs, y_pred_proba, y_pred = _compute_attributions(model, tokenizer, text)
        outputs.append(
            {
                "text": text,
                "true_label": label,
                "pred_label": y_pred,
                "pred_proba": y_pred_proba,
                "tokens": tokens,
                "attributions": attrs,
            }
        )

    # Save a simple TSV per sample
    for i, o in enumerate(outputs, start=1):
        tsv_path = out_dir / f"sample_{i}.tsv"
        df = pd.DataFrame({"token": o["tokens"], "attr": o["attributions"]})
        header_lines = [
            f"# true_label\t{o['true_label']}",
            f"# pred_label\t{o['pred_label']}",
            f"# pred_proba\t{o['pred_proba']:.6f}",
            f"# text\t{o['text']}",
        ]
        tsv_path.write_text("\n".join(header_lines) + "\n", encoding="utf-8")
        with tsv_path.open("a", encoding="utf-8") as f:
            df.to_csv(f, sep="\t", index=False)
    return out_dir


