import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(train_size, valid_size):
    """
    Load positive/negative tweets + test tweets.
    Return:
    - train_texts: list[str]
    - valid_texts: list[str]
    - test_texts: list[str]
    - y_train: list[int]
    - y_valid: list[int]
    """

    # load train (pos + neg)
    with open("twitter-datasets/train_pos_full.txt", "r") as f:
        pos = [line.strip() for line in f]

    with open("twitter-datasets/train_neg_full.txt", "r") as f:
        neg = [line.strip() for line in f]

    train_texts = pos + neg
    train_labels = [1] * len(pos) + [0] * len(neg)

    df = pd.DataFrame({"text": train_texts, "label": train_labels})

    # Split
    df_train, df_valid = train_test_split(
        df, test_size=0.1, random_state=42, shuffle=True
    )

    df_train = df_train.sample(frac=1, random_state=42)  # shuffle

    # Subsample based on requested sizes
    df_train = df_train.sample(train_size, random_state=42)
    df_valid = df_valid.sample(valid_size, random_state=42)

    train_txt = df_train["text"].tolist()
    valid_txt = df_valid["text"].tolist()

    y_train = df_train["label"].tolist()
    y_valid = df_valid["label"].tolist()

    # load test
    test_texts = []
    with open("twitter-datasets/test_data.txt", "r") as f:
        for line in f:
            line = line.strip()
            tweet = line.split(",", 1)[1]  # remove ID
            test_texts.append(tweet)

    return train_txt, valid_txt, test_texts, y_train, y_valid
