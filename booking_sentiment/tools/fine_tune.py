from __future__ import annotations  # ensure forward references in type hints work correctly

import os  # standard library for interacting with the operating system (env vars, paths, etc.)
from pathlib import Path  # object-oriented filesystem paths
from typing import Any, Dict, Tuple  # generic typing helpers

import numpy as np  # numerical operations (not heavily used in this file)
import pandas as pd  # tabular data handling for train/valid/test DataFrames
import torch  # core PyTorch library for tensors and models
from datasets import Dataset, DatasetDict  # Hugging Face datasets wrapper
from sklearn.metrics import matthews_corrcoef  # evaluation metric for binary classification
from torch.nn.functional import softmax  # softmax function for converting logits to probabilities
from transformers import (  # Hugging Face Transformers training and model utilities
    AutoModelForSequenceClassification,  # generic sequence classification model loader
    AutoTokenizer,  # tokenizer loader for the chosen model
    EvalPrediction,  # container for evaluation predictions inside Trainer
    Trainer,  # high-level training loop abstraction
    TrainingArguments,  # configuration object for Trainer
)

from ..config import FineTuneConfig, PathsConfig  # application-specific configuration dataclasses
from ..runtime import configure_runtime  # helper to set seeds, device, and deterministic behavior


class FineTuneModel:
    """
    Encapsulates fine-tuning logic for DistilBERT (or other HF models).
    """

    def __init__(
        self,
        train_df: pd.DataFrame,  # training data containing at least "text" and "label" columns
        valid_df: pd.DataFrame,  # validation data used for model selection
        test_df: pd.DataFrame,  # held-out test data (not used for training)
        fine_tune_cfg: FineTuneConfig,  # hyperparameters and model name configuration
        paths: PathsConfig,  # filesystem paths for saving artifacts
    ) -> None:
        # Store all inputs as instance attributes for later use
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.fine_tune_cfg = fine_tune_cfg
        self.paths = paths

    def _build_datasets(self, tokenizer) -> DatasetDict:
        """
        Convert pandas DataFrames to Hugging Face DatasetDict and tokenize text.
        """
        ds = DatasetDict()  # container for multiple dataset splits
        ds["train"] = Dataset.from_pandas(self.train_df, split="train")  # training split
        ds["valid"] = Dataset.from_pandas(self.valid_df, split="valid")  # validation split
        # test set is not used for training or validation but is prepared for later evaluation
        ds["test"] = Dataset.from_pandas(self.test_df, split="test")  # test split

        def tokenize(examples: dict) -> dict:
            """
            Tokenize a batch of examples and attach the labels.
            """
            # Apply the tokenizer to the "text" column; pad and truncate to model defaults
            encoded = tokenizer(examples["text"], padding=True, truncation=True)
            # Copy labels over so Trainer can use them for supervision
            encoded["label"] = examples["label"]
            return encoded

        # Apply tokenization to all splits in a batched fashion for efficiency
        ds = ds.map(tokenize, batched=True)
        return ds

    def run(self) -> Tuple[Path, Dict[str, Any]]:
        """
        Fine-tune DistilBERT (or other HF model) on GPU or CPU and save artifacts.
        """
        print("[tune] Running fine-tune training")  # simple progress log
        # Directory where the fine-tuned model and related outputs will be stored
        out_dir = self.paths.artifacts_dir / "finetuned_model"
        out_dir.mkdir(parents=True, exist_ok=True)  # create directory tree if needed

        # Determinism settings (seed, device, etc.) for reproducible experiments
        configure_runtime(self.fine_tune_cfg.seed, self.fine_tune_cfg.device)

        # Disable tokenizer parallelism warnings by default unless already set
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Load tokenizer corresponding to the chosen model name
        tokenizer = AutoTokenizer.from_pretrained(self.fine_tune_cfg.model_name)
        # Build and tokenize datasets from DataFrames
        datasets_tokenized = self._build_datasets(tokenizer)
        # Initialize a sequence classification model with 2 labels (binary sentiment)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.fine_tune_cfg.model_name,
            num_labels=2,  # 0 = negative, 1 = positive
        )

        # Compute metrics: MCC (Matthews Correlation Coefficient) for evaluation metrics
        # (MCC is a balanced metric suitable for imbalanced binary classification)
        def compute_metrics(eval_pred: EvalPrediction) -> dict:
            """
            Convert raw model outputs into probabilities, threshold into labels,
            and compute MCC between predictions and ground truth.
            """
            # Flatten true labels array in case of extra dimensions
            y_true = eval_pred.label_ids.ravel()
            # Convert raw logits (numpy) to a PyTorch tensor for softmax
            logits = torch.from_numpy(eval_pred.predictions)
            # Compute probability of the positive class (index 1)
            y_pred_proba = softmax(logits, dim=1)[:, 1].numpy()
            # Apply 0.5 threshold to get binary predictions
            y_pred = (y_pred_proba >= 0.5).astype(int)
            # Compute Matthews Correlation Coefficient as the main metric
            mcc = matthews_corrcoef(y_true, y_pred)
            return {"MCC": mcc}

        # Define all training-related hyperparameters and settings for the Trainer
        training_args = TrainingArguments(
            output_dir=str(out_dir),  # directory for checkpoints and logs
            learning_rate=self.fine_tune_cfg.learning_rate,  # learning rate for optimizer
            num_train_epochs=self.fine_tune_cfg.epochs,  # total number of training epochs
            eval_strategy="steps",  # run evaluation every N steps (see eval_steps)
            save_steps=1000,  # interval for saving checkpoints
            eval_steps=1000,  # interval for running evaluation
            save_total_limit=1,  # keep only the best/most recent checkpoint on disk
            load_best_model_at_end=True,  # restore best-performing checkpoint after training
            seed=self.fine_tune_cfg.seed,  # RNG seed for reproducibility
            data_seed=self.fine_tune_cfg.seed,  # RNG seed for dataset-related shuffling
            fp16=self.fine_tune_cfg.device == "cuda",  # enable mixed precision only on GPU
            dataloader_num_workers=1,  # number of workers for DataLoader (IO parallelism)
            use_cpu=self.fine_tune_cfg.device == "cpu",  # force CPU-only training if requested
        )

        # Create the Trainer object which handles the full training loop
        trainer = Trainer(
            model=model,  # model to fine-tune
            args=training_args,  # training configuration
            train_dataset=datasets_tokenized["train"],  # tokenized training data
            eval_dataset=datasets_tokenized["valid"],  # tokenized validation data
            tokenizer=tokenizer,  # tokenizer for dynamic padding/collation
            compute_metrics=compute_metrics,  # function for computing evaluation metrics
        )
        # Run the training loop according to TrainingArguments
        trainer.train()
        # Save the final/best model weights and configuration to disk
        trainer.save_model(str(out_dir))
        # Save the tokenizer used during training (vocab, merges, config)
        tokenizer.save_pretrained(str(out_dir))
        # Persist tokenized datasets so they can be reused later without recomputation
        tokenized_dir = self.paths.artifacts_dir / "tokenized"
        tokenized_dir.mkdir(parents=True, exist_ok=True)
        datasets_tokenized.save_to_disk(str(tokenized_dir))
        # Return the directory containing the model and the best evaluation metric achieved
        return out_dir, {"best_metric": trainer.state.best_metric}


def fine_tune(
    train_df: pd.DataFrame,  # training data (text + label)
    valid_df: pd.DataFrame,  # validation data
    test_df: pd.DataFrame,  # test data (passed through for completeness)
    fine_tune_cfg: FineTuneConfig,  # hyperparameters and model configuration
    paths: PathsConfig,  # filesystem paths for artifacts
) -> Tuple[Path, Dict[str, Any]]:
    """
    Backwards-compatible wrapper around FineTuneModel.run.
    Creates a FineTuneModel instance and executes the training pipeline.
    """
    # Instantiate the object-oriented fine-tuning helper
    model = FineTuneModel(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        fine_tune_cfg=fine_tune_cfg,
        paths=paths,
    )
    # Launch fine-tuning and return its outputs
    return model.run()