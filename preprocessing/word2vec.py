# preprocessing/word2vec.py

import os
import numpy as np
import pickle
from gensim.models import Word2Vec

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "saved_models/embeddings/word2vec")
EMB_FILE = os.path.join(RESULTS_DIR, "word2vec.model")
VOCAB_FILE = os.path.join(RESULTS_DIR, "vocab.pkl")

# Crée le dossier si nécessaire
def ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"[Word2Vec] Created directory: {RESULTS_DIR}")

# Entraîne ou charge un Word2Vec
def build_word2vec(texts, vector_size=300, window=5, min_count=1, epochs=10):
    ensure_results_dir()

    if os.path.exists(EMB_FILE):
        print("[Word2Vec] Loading existing model...")
        model = Word2Vec.load(EMB_FILE)
    else:
        print("[Word2Vec] Training Word2Vec model...")
        # Tokenize chaque texte en liste de mots
        tokenized = [t.split() for t in texts]
        model = Word2Vec(
            sentences=tokenized,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4
        )
        model.train(tokenized, total_examples=len(tokenized), epochs=epochs)
        model.save(EMB_FILE)
        # save vocab
        with open(VOCAB_FILE, "wb") as f:
            pickle.dump({w: i for i, w in enumerate(model.wv.index_to_key)}, f)
        print(f"[Word2Vec] Model saved to {EMB_FILE}")
    return model

# Convertit un texte en vecteur moyen
def text_to_vec(text, model, vector_size=300):
    tokens = text.split()
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if len(vecs) == 0:
        return np.zeros(vector_size)
    return np.mean(vecs, axis=0)

# Fonction principale utilisée dans la pipeline
def make(train_texts, valid_texts, test_texts):
    # build/load Word2Vec
    all_texts = train_texts + valid_texts + test_texts
    model = build_word2vec(all_texts)
    vector_size = model.vector_size

    # transformer chaque texte en vecteur
    X_train = np.vstack([text_to_vec(t, model, vector_size) for t in train_texts])
    X_valid = np.vstack([text_to_vec(t, model, vector_size) for t in valid_texts])
    X_test  = np.vstack([text_to_vec(t, model, vector_size) for t in test_texts])

    return X_train, X_valid, X_test
