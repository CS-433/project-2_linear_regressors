import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(train_size, valid_size, random_state=42): ##########
    """
    Load positive/negative tweets + test tweets.
    Args:
        train_size (int): Number of training samples
        valid_size (int): Number of validation samples
        random_state (int): Random seed for reproducibility

    Returns:
        train_texts (list[str]): Training set documents
        valid_texts (list[str]): Validation set documents
        test_texts (list[str]): Test set documents
        y_train (list[int]): Training set labels
        y_valid (list[int]): Validation set labels
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
        df, test_size=0.1, random_state=random_state, shuffle=True ##########
    )

    # reset indices
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    # Deterministic subsampling
    df_train = df_train.iloc[:train_size]
    df_valid = df_valid.iloc[:valid_size]

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
