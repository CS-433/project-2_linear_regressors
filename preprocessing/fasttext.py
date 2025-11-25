import numpy as np
from gensim.models import FastText
import os
import pickle

def make(train_txt, valid_txt, test_txt, model_path=None, save_model_path=None, vector_size=100):
    """
    Create FastText embeddings for train, validation, and test sets.

    Args:
        train_txt (list of str): Training tweets
        valid_txt (list of str): Validation tweets
        test_txt (list of str): Test tweets
        model_path (str, optional): Path to pre-trained FastText model
        save_model_path (str, optional): Path to save trained FastText model
        vector_size (int): Embedding size
    
    Returns:
        X_train, X_valid, X_test (np.array): Tweet embeddings
    """

    # Load or train FastText model
    if model_path and os.path.exists(model_path):
        model = FastText.load(model_path)
    else:
        # Train a new FastText model on the training corpus
        sentences = [t.split() for t in train_txt]
        model = FastText(sentences, vector_size=vector_size, window=5, min_count=2)
        if save_model_path:
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            model.save(save_model_path)

    # Function to convert a tweet into a single vector
    def tweet_to_vec(tweet):
        words = tweet.split()
        # Keep only words present in the FastText vocabulary
        vecs = [model.wv[w] for w in words if w in model.wv]
        # Average word vectors, or return zero vector if tweet is empty
        return np.mean(vecs, axis=0) if vecs else np.zeros(vector_size)

    # Convert each dataset to vectors
    X_train = np.array([tweet_to_vec(t) for t in train_txt])
    X_valid = np.array([tweet_to_vec(t) for t in valid_txt])
    X_test = np.array([tweet_to_vec(t) for t in test_txt])

    return X_train, X_valid, X_test
