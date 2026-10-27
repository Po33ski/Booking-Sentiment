from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from captum.attr import InputXGradient
from captum.attr import visualization as viz  # utilities for visualizing attributions
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .inference import MAX_LENGTH, load_preprocessing


class ExplainSamples:
    """
    Compute Input x Gradient attributions for the first N positive and N negative test samples
    and save them as TSV files (plus optional HTML visualizations).
    By default, attributions are computed with respect to the model's predicted class for each sample.
    Optionally, a fixed target class (0 or 1) can be enforced for all samples.
    """

    class ModelWrapper(torch.nn.Module):
        """Runs the model on word embeddings instead of token ids (works for any HF text model)."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, inputs_embeds, attention_mask):
            return self.m(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

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
        # Output directory where per-sample explanation files will be written
        self.out_dir = out_dir
        # Number of positive / negative examples to explain
        self.num_pos = num_pos
        self.num_neg = num_neg
        # Optional fixed target class (0 or 1). If None, the model's predicted class is used per sample.
        self.target = target
        # Whether to save a Captum HTML visualization for each explained sample.
        self.visualize = visualize

    def _visualization_html(
        self,
        tokens: List[str],
        attributions: List[float],
        pred_proba: float,
        pred_label: int,
        true_label: int,
    ) -> str:
        """
        Build a Captum text visualization (colored tokens) for a single example and return it as HTML.
        """
        vis_record = viz.VisualizationDataRecord(
            word_attributions=attributions,
            pred_prob=pred_proba,
            pred_class=pred_label,
            true_class=true_label,
            attr_class=pred_label,
            attr_score=float(sum(attributions)),
            raw_input_ids=tokens,
            convergence_score=0.0,
        )
        # visualize_text also tries to display the HTML; outside a notebook that only
        # prints "<IPython.core.display.HTML object>", so we silence it and save the HTML instead.
        with contextlib.redirect_stdout(io.StringIO()):
            html = viz.visualize_text([vis_record])
        return html.data

    def _compute_attributions(
        self, model, tokenizer, text: str
    ) -> Tuple[List[str], List[float], float, int]:
        """
        Compute Input x Gradient attributions for a single text example.

        - If self.target is None, both the attribution target and reported probability
          correspond to the model's predicted class for this example.
        - If self.target is 0 or 1, both correspond to this fixed class for all examples.
        """
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        special_tokens = encoded.pop("special_tokens_mask")[0].bool()
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        with torch.no_grad():
            probs = softmax(model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=1)[0]

        target_idx = int(torch.argmax(probs).item()) if self.target is None else int(self.target)
        y_pred_proba = float(probs[target_idx].item())

        # Attribute with respect to the word embeddings of the input tokens.
        input_embeddings = model.get_input_embeddings()(input_ids).detach().requires_grad_()
        attributions = InputXGradient(self.ModelWrapper(model)).attribute(
            input_embeddings,
            target=target_idx,
            additional_forward_args=(attention_mask,),
        )

        # One score per token (sum over the embedding dimension), normalized to unit norm.
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / (torch.norm(attributions) + 1e-12)

        # Remove special tokens ([CLS], [SEP], <s>, ...) from both tokens and attributions.
        keep = (~special_tokens).tolist()
        tokens = [tok for tok, k in zip(tokens, keep) if k]
        attributions = [float(a) for a, k in zip(attributions.tolist(), keep) if k]
        return tokens, attributions, y_pred_proba, target_idx

    def run(self) -> Path:
        """
        Compute attributions for the selected samples and write one TSV (and HTML) file per sample.
        """
        out_dir = self.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        model.eval()
        # Apply the same preprocessing as during training.
        lowercase = bool(load_preprocessing(self.model_dir).get("lowercase", False))

        pos_samples = self.test_df[self.test_df["label"] == 1].head(self.num_pos)
        neg_samples = self.test_df[self.test_df["label"] == 0].head(self.num_neg)

        for i, row in enumerate(pd.concat([pos_samples, neg_samples]).itertuples(index=False), start=1):
            text, label = row.text, int(row.label)
            tokens, attrs, y_pred_proba, y_pred = self._compute_attributions(
                model, tokenizer, text.lower() if lowercase else text
            )

            # Save a TSV with a small header and one row per token.
            tsv_path = out_dir / f"sample_{i}.tsv"
            header_lines = [
                f"# true_label\t{label}",
                f"# pred_label\t{y_pred}",
                f"# pred_proba\t{y_pred_proba:.6f}",
                f"# text\t{text}",
            ]
            tsv_path.write_text("\n".join(header_lines) + "\n", encoding="utf-8")
            with tsv_path.open("a", encoding="utf-8") as f:
                pd.DataFrame({"token": tokens, "attr": attrs}).to_csv(f, sep="\t", index=False)

            if self.visualize:
                html = self._visualization_html(tokens, attrs, y_pred_proba, y_pred, label)
                (out_dir / f"sample_{i}.html").write_text(html, encoding="utf-8")
        return out_dir


def explain_samples(
    model_dir: Path,
    test_df: pd.DataFrame,
    out_dir: Path,
    num_pos: int = 3,
    num_neg: int = 3,
    target: Optional[int] = None,
    visualize: bool = False,
) -> Path:
    """
    Convenience function used by the CLI: computes the explanations and returns the output directory.
    If target is None, explanations are computed with respect to the model's predicted class
    for each sample; otherwise they target the fixed class index provided here (0 or 1).
    If visualize is True, an HTML visualization is saved next to each TSV file.
    """
    return ExplainSamples(
        model_dir=model_dir,
        test_df=test_df,
        out_dir=out_dir,
        num_pos=num_pos,
        num_neg=num_neg,
        target=target,
        visualize=visualize,
    ).run()
