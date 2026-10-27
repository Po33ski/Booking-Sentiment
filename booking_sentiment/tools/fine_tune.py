from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

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

from ..config import FineTuneConfig, PathsConfig
from ..runtime import configure_runtime, resolve_device
from .inference import MAX_LENGTH, save_preprocessing


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """
    Convert raw model outputs into probabilities, threshold at 0.5 and compute MCC
    (a balanced metric that is also used to pick the best checkpoint).
    """
    y_true = eval_pred.label_ids.ravel()
    logits = torch.from_numpy(eval_pred.predictions).float()
    y_pred_proba = softmax(logits, dim=1)[:, 1].numpy()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return {"MCC": matthews_corrcoef(y_true, y_pred)}


class FineTuneModel:
    """
    Fine-tunes DistilBERT (or another HF text model) for binary sentiment classification.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,  # training data with "text" and "label" columns
        valid_df: pd.DataFrame,  # validation data used for model selection
        fine_tune_cfg: FineTuneConfig,
        paths: PathsConfig,
        lowercase: bool = False,  # whether the training texts were lowercased by the clean step
    ) -> None:
        self.train_df = train_df
        self.valid_df = valid_df
        self.fine_tune_cfg = fine_tune_cfg
        self.paths = paths
        self.lowercase = lowercase

    def _build_datasets(self, tokenizer) -> DatasetDict:
        """
        Convert pandas DataFrames to a tokenized Hugging Face DatasetDict.
        Padding is done per batch by the Trainer's data collator.
        """
        ds = DatasetDict(
            {
                "train": Dataset.from_pandas(self.train_df[["text", "label"]], preserve_index=False),
                "valid": Dataset.from_pandas(self.valid_df[["text", "label"]], preserve_index=False),
            }
        )

        def tokenize(examples: dict) -> dict:
            return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

        return ds.map(tokenize, batched=True, remove_columns=["text"])

    def run(self) -> Tuple[Path, Dict[str, Any]]:
        """
        Fine-tune the model on GPU or CPU and save the best checkpoint (by validation MCC).
        """
        print("[tune] Running fine-tune training")
        cfg = self.fine_tune_cfg
        model_dir = self.paths.artifacts_dir / "finetuned_model"
        # Intermediate checkpoints live outside the model directory and are removed after training.
        checkpoints_dir = self.paths.artifacts_dir / "checkpoints"
        # Leftovers from an interrupted run would be mixed with the new checkpoints.
        shutil.rmtree(checkpoints_dir, ignore_errors=True)

        configure_runtime(cfg.seed, cfg.device)
        use_cuda = resolve_device(cfg.device) == "cuda"

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        datasets_tokenized = self._build_datasets(tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=2,
            id2label={0: "negative", 1: "positive"},
            label2id={"negative": 0, "positive": 1},
        )

        training_args = TrainingArguments(
            output_dir=str(checkpoints_dir),
            learning_rate=cfg.learning_rate,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size * 2,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            eval_strategy="epoch",  # evaluate after every epoch (works for any dataset size)
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,  # restore the best checkpoint after training...
            metric_for_best_model="MCC",  # ...where "best" means the highest validation MCC
            greater_is_better=True,
            seed=cfg.seed,
            data_seed=cfg.seed,
            fp16=use_cuda,  # mixed precision only when a GPU is actually used
            use_cpu=not use_cuda,
            dataloader_num_workers=1,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets_tokenized["train"],
            eval_dataset=datasets_tokenized["valid"],
            processing_class=tokenizer,  # also enables dynamic padding
            compute_metrics=compute_metrics,
        )
        trainer.train()

        # Save only the final (best) model + tokenizer into a fresh directory.
        if model_dir.exists():
            shutil.rmtree(model_dir)
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        save_preprocessing(model_dir, lowercase=self.lowercase)
        shutil.rmtree(checkpoints_dir, ignore_errors=True)

        print(f"[tune] Best validation MCC: {trainer.state.best_metric}")
        return model_dir, {"best_metric": trainer.state.best_metric}


def fine_tune(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    fine_tune_cfg: FineTuneConfig,
    paths: PathsConfig,
    lowercase: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Convenience wrapper used by the CLI.
    """
    return FineTuneModel(
        train_df=train_df,
        valid_df=valid_df,
        fine_tune_cfg=fine_tune_cfg,
        paths=paths,
        lowercase=lowercase,
    ).run()
