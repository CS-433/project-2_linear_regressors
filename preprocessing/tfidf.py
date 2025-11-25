from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def make(train_txt, valid_txt, test_txt, max_features=50_000, ngram_range=(1,2), save_path=None):
    """
    Create TF-IDF representations for train, validation, and test sets.

    Args:
        train_txt (list of str): Training tweets
        valid_txt (list of str): Validation tweets
        test_txt (list of str): Test tweets
        max_features (int): Maximum number of features (words/ngrams)
        ngram_range (tuple): Min and max n-gram size
        save_path (str, optional): Path to save the fitted vectorizer
    
    Returns:
        X_train, X_valid, X_test (sparse matrices): TF-IDF features
    """

    #initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=True,
        strip_accents='unicode'
    )

    #fit on training data and transform
    X_train = vectorizer.fit_transform(train_txt)

    #transform validation and test sets
    X_valid = vectorizer.transform(valid_txt)
    X_test = vectorizer.transform(test_txt)

    #optionally save the vectorizer for future use
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(vectorizer, f)

    return X_train, X_valid, X_test
