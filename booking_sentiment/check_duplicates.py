import pandas as pd
from IPython.display import display


def check_duplicates(df_neg: pd.DataFrame, df_pos: pd.DataFrame, df_raw: pd.DataFrame):
    """
    Check for duplicates in the dataframe and return a dataframe with columns: text, label
    """
    print("Checking for duplicates in the fresh dataframe")
    print("\n -- Negative reviews: ")
    neg_value_counts = df_neg.value_counts()
    total_neg_reviews = len(df_neg)

    top_20_neg = pd.DataFrame({
        'Count': neg_value_counts.head(20),
        'Percentage': (neg_value_counts.head(20) / total_neg_reviews * 100).round(2)
    })
    display(top_20_neg)

    print("\n -- Positive reviews: ")
    pos_value_counts = df_pos.value_counts()
    total_pos_reviews = len(df_pos)

    top_20_pos = pd.DataFrame({
        'Count': pos_value_counts.head(20),
        'Percentage': (pos_value_counts.head(20) / total_pos_reviews * 100).round(2)
    })
    display(top_20_pos)

    print("\n -- Whole Dataset: ")
    all_value_counts = df_raw.value_counts()
    total_all_reviews = len(df_raw)

    top_20_all = pd.DataFrame({
        'Count': all_value_counts.head(20),
        'Percentage': (all_value_counts.head(20) / total_all_reviews * 100).round(2)
    })
    display(top_20_all)

    return None