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


class ExplainSamples:
    """
    Compute InputXGradient attributions for first N positive and N negative samples and save TSVs.
    """

    class ModelWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, inputs, attention_mask):
            return self.m(inputs, attention_mask=attention_mask)[0]

    def __init__(
        self,
        model_dir: Path,
        test_df: pd.DataFrame,
        out_dir: Path,
        num_pos: int = 3,
        num_neg: int = 3,
    ) -> None:
        self.model_dir = model_dir
        self.test_df = test_df
        self.out_dir = out_dir
        self.num_pos = num_pos
        self.num_neg = num_neg

    def _compute_attributions(
        self, model, tokenizer, text: str
    ) -> Tuple[List[str], List[float], float, int]:
        # encode the text using the tokenizer. This returns a dictionary with the input_ids, attention_mask, and token_type_ids.
        tokenizer_output = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        # convert the input_ids to tokens. This returns a list of tokens.
        tokens = tokenizer.convert_ids_to_tokens(tokenizer_output["input_ids"][0])

        model_copy = deepcopy(model)
        # configure the interpretable embedding layer. This returns a callable that can be used to get the embeddings for a given input.
        interpretable_embedding_layer = configure_interpretable_embedding_layer(
            model_copy, "distilbert.embeddings"
        )
        # get the embeddings for the input. This returns a tensor of shape (batch_size, sequence_length, embedding_dim).
        input_embeddings = interpretable_embedding_layer.indices_to_embeddings(
            tokenizer_output["input_ids"]
        )

        model_wrapper = self.ModelWrapper(model_copy)  # wrap the model in a callable that can be used to get the embeddings for a given input.
        input_x_gradient = InputXGradient(
            model_wrapper
        )  # create an InputXGradient object that can be used to compute the attributions.
        attributions = input_x_gradient.attribute(  # compute the attributions. This returns a tensor of shape (batch_size, sequence_length, embedding_dim).
            input_embeddings,
            target=1,
            additional_forward_args=tokenizer_output["attention_mask"],
        )

        attributions = attributions.sum(
            dim=-1
        ).squeeze(0)  # sum the attributions over the embedding dimension. This returns a tensor of shape (batch_size, sequence_length).
        norm = torch.norm(attributions)
        attributions = attributions / (
            norm + 1e-12
        )  # normalize the attributions. This returns a tensor of shape (batch_size, sequence_length).
        attributions = list(
            [float(a) for a in attributions]
        )  # convert the attributions to a list of floats. This returns a list of floats.

        with torch.no_grad():
            logits = model(
                **tokenizer_output
            ).logits  # get the logits for the input. This returns a tensor of shape (batch_size, num_labels).
            y_pred_proba = softmax(
                logits, dim=1
            )[0, 1].item()  # get the probability of the positive class. This returns a float.
            y_pred = int(
                y_pred_proba >= 0.5
            )  # get the predicted class. This returns an integer.

        # remove special tokens
        tokens = tokens[
            1:-1
        ]  # remove the special tokens. This returns a list of tokens.
        attributions = attributions[
            1:-1
        ]  # remove the special tokens. This returns a list of attributions. This returns a list of floats.
        return tokens, attributions, y_pred_proba, y_pred

    def run(self) -> Path:
        """
        Compute InputXGradient attributions for first N positive and N negative samples and save TSVs.
        """
        out_dir = self.out_dir
        model_dir = self.model_dir
        test_df = self.test_df
        num_pos = self.num_pos
        num_neg = self.num_neg

        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()

        pos_samples = test_df[
            test_df["label"] == 1
        ].head(num_pos)  # get the first num_pos positive samples. This returns a pandas DataFrame.
        neg_samples = test_df[
            test_df["label"] == 0
        ].head(num_neg)  # get the first num_neg negative samples. This returns a pandas DataFrame.

        outputs = []
        for row in pd.concat(
            [pos_samples, neg_samples]
        ).itertuples(
            index=False
        ):  # concatenate the positive and negative samples. This returns a pandas DataFrame.
            text, label = row.text, int(row.label)
            tokens, attrs, y_pred_proba, y_pred = self._compute_attributions(
                model, tokenizer, text
            )
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

        # Save a simple TSV per sample. This saves the tokens and attributions for each sample in a TSV file.
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


def explain_samples(model_dir: Path, test_df: pd.DataFrame, out_dir: Path, num_pos: int = 3, num_neg: int = 3) -> ExplainSamples:
    """
    Convenience factory used by the CLI.
    """
    return ExplainSamples(model_dir, test_df, out_dir, num_pos, num_neg)