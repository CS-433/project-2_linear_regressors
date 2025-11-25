# preprocessing/word2vec.py
import os
import numpy as np
import pickle
from gensim.models import Word2Vec

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "saved_models/embeddings/word2vec")
EMB_FILE = os.path.join(RESULTS_DIR, "word2vec.model")
VOCAB_FILE = os.path.join(RESULTS_DIR, "vocab.pkl")


def ensure_results_dir():
    """
    Ensure that the directory used to store Word2Vec artifacts exists.
    Creates the directory if it does not already exist.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"[Word2Vec] Created directory: {RESULTS_DIR}")


def build_word2vec(texts, vector_size=300, window=5, min_count=1, epochs=10):
    """
    Train or load a Word2Vec embedding model.

    If a saved model already exists, it is loaded. Otherwise, a new
    Word2Vec model is trained on the provided texts and saved to disk.

    Parameters
    ----------
    texts : list of str
        List of raw text documents.
    vector_size : int, optional (default=300)
        Dimensionality of each word embedding vector.
    window : int, optional (default=5)
        Context window size used by Word2Vec.
    min_count : int, optional (default=1)
        Minimum word frequency required to be included in the vocabulary.
    epochs : int, optional (default=10)
        Number of training epochs.

    Returns
    -------
    Word2Vec
        Trained or loaded Word2Vec model.
    """
    ensure_results_dir()

    if os.path.exists(EMB_FILE):
        print("[Word2Vec] Loading existing model...")
        model = Word2Vec.load(EMB_FILE)
    else:
        print("[Word2Vec] Training Word2Vec model...")

        # Split each text into tokens
        tokenized = [t.split() for t in texts]

        # Initialize and train the model
        model = Word2Vec(
            sentences=tokenized,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4
        )
        model.train(tokenized, total_examples=len(tokenized), epochs=epochs)

        # Save model
        model.save(EMB_FILE)

        # Save vocabulary: word -> index
        with open(VOCAB_FILE, "wb") as f:
            pickle.dump({w: i for i, w in enumerate(model.wv.index_to_key)}, f)

        print(f"[Word2Vec] Model saved to {EMB_FILE}")

    return model


def text_to_vec(text, model, vector_size=300):
    """
    Convert a raw text into a single embedding vector by averaging
    the embeddings of all known words.

    Unknown words (not in the Word2Vec vocabulary) are ignored.

    Parameters
    ----------
    text : str
        Input sentence/document.
    model : Word2Vec
        Word2Vec model providing word embeddings.
    vector_size : int, optional (default=300)
        Dimensionality of the output vector.

    Returns
    -------
    numpy.ndarray, shape (vector_size,)
        Averaged embedding for the text. Returns a zero vector
        if no word in the text is found in the model.
    """
    tokens = text.split()
    vecs = [model.wv[w] for w in tokens if w in model.wv]

    if len(vecs) == 0:
        return np.zeros(vector_size)

    return np.mean(vecs, axis=0)


def make(train_texts, valid_texts, test_texts):
    """
    Main preprocessing function for the pipeline.
    Builds (or loads) a Word2Vec model on all provided texts, and
    converts each document into a fixed-size embedding vector.

    Parameters
    ----------
    train_texts : list of str
        Training set documents.
    valid_texts : list of str
        Validation set documents.
    test_texts : list of str
        Test set documents.

    Returns
    -------
    X_train : numpy.ndarray, shape (n_train, vector_size)
        Matrix of averaged embeddings for the training texts.
    X_valid : numpy.ndarray, shape (n_valid, vector_size)
        Matrix of averaged embeddings for the validation texts.
    X_test : numpy.ndarray, shape (n_test, vector_size)
        Matrix of averaged embeddings for the test texts.
    """
    # Build or load the model using all available texts
    all_texts = train_texts + valid_texts + test_texts
    model = build_word2vec(all_texts)
    vector_size = model.vector_size

    # Convert each text to a vector
    X_train = np.vstack([text_to_vec(t, model, vector_size) for t in train_texts])
    X_valid = np.vstack([text_to_vec(t, model, vector_size) for t in valid_texts])
    X_test = np.vstack([text_to_vec(t, model, vector_size) for t in test_texts])

    return X_train, X_valid, X_test
