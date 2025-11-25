from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef
from torch.nn.functional import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from .config import TrainConfig, PathsConfig


def _build_datasets(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, tokenizer) -> DatasetDict:
    ds = DatasetDict()
    ds["train"] = Dataset.from_pandas(train_df, split="train")
    ds["valid"] = Dataset.from_pandas(valid_df, split="valid")
    ds["test"] = Dataset.from_pandas(test_df, split="test") # test set is not used for training or validation
    def tokenize(examples: dict) -> dict:
        encoded = tokenizer(examples["text"], padding=True, truncation=True)
        encoded["label"] = examples["label"]
        return encoded

    ds = ds.map(tokenize, batched=True)
    return ds


def train_hf(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, train_cfg: TrainConfig, paths: PathsConfig) -> Tuple[Path, Dict[str, Any]]:
    """
    Fine-tune DistilBERT (or other HF model) on GPU or CPU and save artifacts.
    """
    print("[train] Running fine-tune training")
    out_dir = paths.artifacts_dir / "finetuned_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determinism settings
    os.environ["PYTHONHASHSEED"] = str(train_cfg.seed)
    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    if train_cfg.device == "cuda":
        torch.manual_seed(train_cfg.seed)
        torch.cuda.manual_seed_all(train_cfg.seed)
    else:
        torch.manual_seed(train_cfg.seed)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Load tokenizer and datasets    
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name)
    # Build datasets
    datasets_tokenized = _build_datasets(train_df, valid_df, test_df, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(train_cfg.model_name, num_labels=2) # num_labels=2 because we have 2 classes: positive and negative

    # Compute metrics: MCC (Matthews Correlation Coefficient) for evaluation metrics (MCC is a measure of the quality of the classification model)
    def compute_metrics(eval_pred: EvalPrediction) -> dict:
        y_true = eval_pred.label_ids.ravel()
        logits = torch.from_numpy(eval_pred.predictions)
        y_pred_proba = softmax(logits, dim=1)[:, 1].numpy()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        return {"MCC": mcc}

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=train_cfg.learning_rate,
        num_train_epochs=train_cfg.epochs,
        eval_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=1,
        load_best_model_at_end=True,
        seed=train_cfg.seed,
        data_seed=train_cfg.seed,
        fp16=train_cfg.device == "cuda",
        dataloader_num_workers=1, # the previous checkpoint is deleted
        use_cpu= train_cfg.device == "cpu"
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets_tokenized["train"],
        eval_dataset=datasets_tokenized["valid"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    # Train the model
    trainer.train()
    # Save the model
    trainer.save_model(str(out_dir))
    # Save the tokenizer
    
    tokenizer.save_pretrained(str(out_dir))
    # Save the tokenized datasets
    tokenized_dir = paths.artifacts_dir / "tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    datasets_tokenized.save_to_disk(str(tokenized_dir))
    # Return the output directory and the best metric
    return out_dir, {"best_metric": trainer.state.best_metric}


