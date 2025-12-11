from __future__ import annotations

from copy import deepcopy  # used to create a copy of the model for interpretable embeddings
from pathlib import Path  # filesystem paths for model directory and outputs
from typing import Iterable, List, Tuple, Optional  # basic typing helpers

import numpy as np
import pandas as pd
import torch
from captum.attr import InputXGradient, configure_interpretable_embedding_layer
from captum.attr import visualization as viz  # utilities for visualizing attributions
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ExplainSamples:
    """
    Compute InputXGradient attributions for first N positive and N negative samples and save TSVs.
    By default, attributions are computed with respect to the model's predicted class for each sample.
    Optionally, a fixed target class (0 or 1) can be enforced for all samples.
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
        target: Optional[int] = None,
        visualize: bool = False,
    ) -> None:
        # Directory containing the fine-tuned Hugging Face model
        self.model_dir = model_dir
        # Full test DataFrame from which we will select examples to explain
        self.test_df = test_df
        # Output directory where per-sample TSV explanation files will be written
        self.out_dir = out_dir
        # Number of positive examples to explain
        self.num_pos = num_pos
        # Number of negative examples to explain
        self.num_neg = num_neg
        # Optional fixed target class (0 or 1). If None, the model's predicted class is used per sample.
        self.target = target
        # Whether to render Captum text visualizations for each explained sample.
        self.visualize = visualize

    def _visualize_sample(
        self,
        tokens: List[str],
        attributions: List[float],
        pred_proba: float,
        pred_label: int,
        true_label: int,
    ) -> None:
        """
        Render a Captum text visualization for a single example.

        This mirrors the pattern used in the LAB_INSTRUCTION notebook, using
        VisualizationDataRecord and visualize_text to display colored tokens.
        """
        # Use the explained class (pred_label) as the attribution class for the legend.
        attr_class = pred_label
        # Aggregate attribution scores for a simple scalar summary.
        attr_score = float(sum(attributions))

        vis_record = viz.VisualizationDataRecord(
            word_attributions=attributions,
            pred_prob=pred_proba,
            pred_class=pred_label,
            true_class=true_label,
            attr_class=attr_class,
            attr_score=attr_score,
            raw_input_ids=tokens,
            convergence_score=0.0,
        )
        # Display the visualization. In a notebook this renders HTML;
        # in other environments, this may fall back to a text representation.
        viz.visualize_text([vis_record])

    def _compute_attributions(
        self, model, tokenizer, text: str
    ) -> Tuple[List[str], List[float], float, int]:
        """
        Compute Input×Gradient attributions for a single text example.

        - If self.target is None, both the attribution target and reported probability
          correspond to the model's predicted class for this example.
        - If self.target is 0 or 1, both the attribution target and reported probability
          correspond to this fixed class for all examples.
        """
        # Encode the text using the tokenizer. This returns a dictionary with
        # "input_ids" and "attention_mask" tensors on CPU.
        tokenizer_output = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        # Convert the input_ids back to tokens for later visualization.
        tokens = tokenizer.convert_ids_to_tokens(tokenizer_output["input_ids"][0])

        with torch.no_grad():
            # Forward pass through the original model to obtain logits of shape
            # (batch_size=1, num_labels=2).
            logits = model(**tokenizer_output).logits
            # Convert logits to probabilities with softmax along the class dimension.
            probs = softmax(logits, dim=1)[0]  # shape: (num_labels,)

        # Decide which class to explain:
        # - if self.target is None: use the model's predicted class (argmax over probs)
        # - otherwise: use the fixed class provided at construction time.
        if self.target is None:
            target_idx = int(torch.argmax(probs).item())
        else:
            target_idx = int(self.target)

        # Probability associated with the chosen target class.
        y_pred_proba = float(probs[target_idx].item())
        # Predicted label is the target class we are explaining.
        y_pred = target_idx

        # Create a copy of the model for the interpretable embedding layer,
        # so that Captum can operate on embeddings directly.
        model_copy = deepcopy(model)
        # Configure the interpretable embedding layer. This returns a callable
        # that can be used to get the embeddings for a given input.
        interpretable_embedding_layer = configure_interpretable_embedding_layer(
            model_copy, "distilbert.embeddings"
        )
        # Get the embeddings for the input. This returns a tensor of shape
        # (batch_size, sequence_length, embedding_dim).
        input_embeddings = interpretable_embedding_layer.indices_to_embeddings(
            tokenizer_output["input_ids"]
        )

        # Wrap the model in a module that accepts embeddings instead of token IDs.
        model_wrapper = self.ModelWrapper(model_copy)
        # Create an InputXGradient object that can be used to compute the attributions.
        input_x_gradient = InputXGradient(model_wrapper)
        # Compute the attributions with respect to the chosen target class.
        # The result has shape (batch_size, sequence_length, embedding_dim).
        attributions = input_x_gradient.attribute(
            input_embeddings,
            target=target_idx,
            additional_forward_args=tokenizer_output["attention_mask"],
        )

        # Sum the attributions over the embedding dimension to obtain a single
        # score per token, then squeeze the batch dimension.
        attributions = attributions.sum(dim=-1).squeeze(0)
        # Normalize the attribution vector to have unit norm for readability.
        norm = torch.norm(attributions)
        attributions = attributions / (norm + 1e-12)
        # Convert the tensor of attributions to a plain Python list of floats.
        attributions = [float(a) for a in attributions]

        # Remove special tokens ([CLS], [SEP]) from both tokens and attributions.
        tokens = tokens[1:-1]
        attributions = attributions[1:-1]
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
            # Compute attributions for this text. Depending on self.target,
            # this will either explain the model's predicted class (default)
            # or a fixed global target class.
            tokens, attrs, y_pred_proba, y_pred = self._compute_attributions(
                model, tokenizer, text
            )
            # Optionally visualize the attributions using Captum's HTML/text viewer.
            if self.visualize:
                self._visualize_sample(
                    tokens=tokens,
                    attributions=attrs,
                    pred_proba=y_pred_proba,
                    pred_label=y_pred,
                    true_label=label,
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


def explain_samples(
    model_dir: Path,
    test_df: pd.DataFrame,
    out_dir: Path,
    num_pos: int = 3,
    num_neg: int = 3,
    target: Optional[int] = None,
    visualize: bool = False,
) -> ExplainSamples:
    """
    Convenience factory used by the CLI.
    If target is None, explanations are computed with respect to the model's
    predicted class for each sample; otherwise, explanations target the fixed
    class index provided here (0 or 1).

    If visualize is True, Captum text visualizations are rendered for each sample
    in addition to writing TSV files.
    """
    return ExplainSamples(
        model_dir=model_dir,
        test_df=test_df,
        out_dir=out_dir,
        num_pos=num_pos,
        num_neg=num_neg,
        target=target,
        visualize=visualize,
    )