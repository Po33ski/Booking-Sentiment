import pandas as pd

from booking_sentiment.config import SplitConfig
from booking_sentiment.tools.splits import sample_dataframe, split_dataframe


def _df(n_neg=400, n_pos=600):
    return pd.DataFrame({"text": [f"t{i}" for i in range(n_neg + n_pos)], "label": [0] * n_neg + [1] * n_pos})


def test_sample_is_stratified():
    sampled = sample_dataframe(_df(), sample_size=100, random_state=0)
    assert len(sampled) == 100
    assert sampled["label"].mean() == 0.6


def test_sample_larger_than_data_uses_all_rows():
    assert len(sample_dataframe(_df(), sample_size=10_000, random_state=0)) == 1000
    assert len(sample_dataframe(_df(), sample_size=None, random_state=0)) == 1000


def test_splits_are_disjoint_and_sized():
    train, valid, test = split_dataframe(_df(), SplitConfig())
    assert len(test) == 300 and len(train) + len(valid) == 700
    assert abs(len(valid) - 100) <= 1  # sklearn rounds the split size up
    texts = [set(p["text"]) for p in (train, valid, test)]
    assert not (texts[0] & texts[1]) and not (texts[0] & texts[2]) and not (texts[1] & texts[2])
