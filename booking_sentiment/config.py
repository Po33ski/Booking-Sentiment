from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    # Unknown keys (e.g. a typo like "learning_rte") raise an error instead of being ignored.
    model_config = ConfigDict(extra="forbid")


# DatasetConfig: Configuration for the dataset
class DatasetConfig(_StrictModel):
    dataset_name: str = Field(default="morgul10/booking_reviews")
    # positive_col: name of the positive column from the dataset
    positive_col: str = Field(default="Positive_Review")
    # negative_col: name of the negative column from the dataset
    negative_col: str = Field(default="Negative_Review")
    # sample_size: number of rows sampled (stratified) from the cleaned data before splitting; null = use all
    sample_size: Optional[int] = Field(default=10000, ge=1)
    # random_state: random state used for sampling
    random_state: int = Field(default=0)


# CleaningConfig: Configuration for the cleaning like removing terms, casefolding, deduplication, etc.
class CleaningConfig(_StrictModel):
    # remove_terms: texts equal to one of these terms (case-insensitive) are dropped
    remove_terms: List[str] = Field(
        default_factory=lambda: [
            "No Negative",
            "No Positive",
            "nothing",
            "nothing really",
            "none",
            "n a",
            "na",
            "everything",
            "location",
            "the location",
            "breakfast",
            "the breakfast",
            "staff",
        ]
    )
    # casefold_dedup: case-insensitive deduplication
    casefold_dedup: bool = Field(default=True)
    # min_text_len: minimum text length (empty texts are always dropped)
    min_text_len: int = Field(default=1, ge=1)
    # lowercase_text: lowercase the text
    lowercase_text: bool = Field(default=True)


# SplitConfig: Configuration for the split
class SplitConfig(_StrictModel):
    train_frac: float = Field(default=0.6, gt=0.0, lt=1.0)
    valid_frac: float = Field(default=0.1, gt=0.0, lt=1.0)
    test_frac: float = Field(default=0.3, gt=0.0, lt=1.0)
    random_state: int = Field(default=0)
    stratify: bool = Field(default=True)

    # The split fractions must add up to 1.0
    @model_validator(mode="after")
    def _check_sum(self) -> "SplitConfig":
        total = self.train_frac + self.valid_frac + self.test_frac
        if abs(total - 1.0) > 1e-9:
            raise ValueError("train_frac + valid_frac + test_frac must equal 1.0")
        return self


# FineTuneConfig: Configuration for the training like model name, learning rate, epochs, seed, device, etc.
class FineTuneConfig(_StrictModel):
    model_name: str = Field(default="distilbert/distilbert-base-uncased")
    learning_rate: float = Field(default=2e-5, gt=0.0)
    epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=16, ge=1)
    warmup_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    seed: int = Field(default=0)
    # device: falls back to CPU (with a warning) when CUDA is requested but not available
    device: Literal["cuda", "cpu"] = Field(default="cuda")


# QualityConfig: Configuration for the quality step like embedding model name, cv folds, logistic C, etc.
class QualityConfig(_StrictModel):
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    cv_folds: int = Field(default=5, ge=2)
    regularization_c: float = Field(default=0.1, gt=0.0)
    # device: device to use for the embedding model
    device: Literal["cuda", "cpu"] = Field(default="cuda")
    # iterations: number of Cleanlab passes over train+valid
    iterations: int = Field(default=1, ge=1)
    # test_leak_similarity: train/valid rows with cosine similarity >= this value to any test row are dropped
    test_leak_similarity: float = Field(default=0.95, gt=0.0, le=1.0)


# PathsConfig: Configuration for the paths like artifacts dir, etc.
class PathsConfig(_StrictModel):
    artifacts_dir: Path = Field(default=Path("artifacts"))


# ProjectConfig: Configuration for the project like dataset, cleaning, split, fine_tune, quality, paths, etc.
class ProjectConfig(_StrictModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    fine_tune: FineTuneConfig = Field(default_factory=FineTuneConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    # load: load the configuration from a JSON file or return the default configuration
    @staticmethod
    def load(config_path: Optional[str]) -> "ProjectConfig":
        if config_path is None:
            return ProjectConfig()
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Config file must be JSON; parse error: {e}") from e
        return ProjectConfig.model_validate(data)
