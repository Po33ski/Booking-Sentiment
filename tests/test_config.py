import json

import pytest
from pydantic import ValidationError

from booking_sentiment.config import ProjectConfig, SplitConfig


def test_quick_config_loads():
    cfg = ProjectConfig.load("configs/quick.json")
    assert cfg.fine_tune.learning_rate == pytest.approx(2e-5)


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"fine_tune": {"learning_rte": 0.1}}))
    with pytest.raises(ValidationError):
        ProjectConfig.load(str(path))


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValidationError):
        SplitConfig(train_frac=0.5, valid_frac=0.1, test_frac=0.3)


def test_zero_split_fraction_is_rejected():
    with pytest.raises(ValidationError):
        SplitConfig(train_frac=0.7, valid_frac=0.0, test_frac=0.3)
