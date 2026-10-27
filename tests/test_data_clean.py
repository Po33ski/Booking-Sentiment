import pandas as pd

from booking_sentiment.config import CleaningConfig
from booking_sentiment.tools.data_clean import clean_and_label


def test_conflicting_labels_and_terms_are_removed():
    neg = pd.Series(["No Negative", "Toiletries", "dirty room", "", None, "a.b"])
    pos = pd.Series(["toiletries", "Great staff", "No Positive", "aXb"])
    cfg = CleaningConfig(remove_terms=["No Negative", "No Positive", "a.b"])

    df = clean_and_label(neg, pos, cfg)

    # "toiletries" appears with both labels -> dropped; empty/None dropped; "a.b" matched literally
    assert sorted(df["text"]) == ["axb", "dirty room", "great staff"]
    assert dict(zip(df["text"], df["label"])) == {"axb": 1, "dirty room": 0, "great staff": 1}


def test_empty_remove_terms_keeps_texts():
    df = clean_and_label(pd.Series(["bad room"]), pd.Series(["nice hotel"]), CleaningConfig(remove_terms=[]))
    assert len(df) == 2


def test_casefold_dedup_without_lowercasing():
    cfg = CleaningConfig(remove_terms=[], lowercase_text=False, casefold_dedup=True)
    df = clean_and_label(pd.Series(["Bad Room", "bad room"]), pd.Series(["nice"]), cfg)
    assert (df["label"] == 0).sum() == 1
