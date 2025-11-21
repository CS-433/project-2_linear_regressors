import numpy as np
import pickle
import os
import subprocess
import shutil

# Path depending of file location
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
GLOVE_DIR = os.path.join(BASE_DIR, "glove_scripts")
RESULTS_DIR = os.path.join(GLOVE_DIR, "results")

VOCAB_PKL = os.path.join(RESULTS_DIR, "vocab.pkl")
COOC_PKL = os.path.join(RESULTS_DIR, "cooc.pkl")
EMB_NPY = os.path.join(RESULTS_DIR, "embeddings.npy")

# ensure results dir
def ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"[GLOVE] Created directory: {RESULTS_DIR}")

# Move GloVe outputs to glove_scripts/results/
def move_glove_outputs():
    outputs = [
        "vocab.pkl",
        "vocab_cut.txt",
        "vocab_full.txt",
        "cooc.pkl",
        "embeddings.npy",
    ]

    for fname in outputs:
        src = os.path.join(GLOVE_DIR, fname)
        dst = os.path.join(RESULTS_DIR, fname)

        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"[GLOVE] Moved {fname} → results/")


# Run GloVe pipeline if needed
def build_glove_if_needed():
    """
    Run GloVe scripts to build embeddings.npy if not already existing.
    """
    if os.path.exists(EMB_NPY):
        print("[GLOVE] embeddings.npy already exists → skipping GloVe build.")
        return

    print("[GLOVE] running GloVe pipeline...")

    # Run exact scripts (in glove_scripts/)
    subprocess.run(["bash", "glove_scripts/build_vocab.sh"], cwd=BASE_DIR, check=True)
    subprocess.run(["bash", "glove_scripts/cut_vocab.sh"], cwd=BASE_DIR, check=True)
    subprocess.run(["python", "glove_scripts/pickle_vocab.py"], cwd=BASE_DIR, check=True)
    subprocess.run(["python", "glove_scripts/cooc.py"], cwd=BASE_DIR, check=True)
    subprocess.run(["python", "glove_scripts/glove_solution.py"], cwd=BASE_DIR, check=True)

    # Move outputs into glove_scripts/results/
    ensure_results_dir()
    outputs = ["vocab.pkl", "vocab_cut.txt", "vocab_full.txt", "cooc.pkl", "embeddings.npy"]
    for name in outputs:
        src = os.path.join(BASE_DIR, name)
        dst = os.path.join(RESULTS_DIR, name)
        if os.path.exists(src):
            shutil.move(src, dst)

    print("[GLOVE] Training complete, embeddings generated in glove_scripts/results/.")


# load GloVe embeddings
def load_glove():
    """
    Load vocab.pkl + embeddings.npy and return a dict {word: vector}.
    """
    with open(VOCAB_PKL, "rb") as f:
        vocab = pickle.load(f)

    emb_matrix = np.load(EMB_NPY)
    dim = emb_matrix.shape[1]

    embeddings = {word: emb_matrix[idx] for word, idx in vocab.items()}

    return embeddings, dim


# text to vector
def text_to_vec(text, embeddings, dim):
    tokens = text.split()
    vecs = [embeddings[t] for t in tokens if t in embeddings]

    if len(vecs) == 0:
        return np.zeros(dim)

    return np.mean(vecs, axis=0)


# make function
def make(train_texts, valid_texts, test_texts):
    # 1) Build GloVe if needed
    build_glove_if_needed()

    # 2) Load embeddings
    embeddings, dim = load_glove()

    # 3) Convert each tweet to vector
    X_train = np.vstack([text_to_vec(t, embeddings, dim) for t in train_texts])
    X_valid = np.vstack([text_to_vec(t, embeddings, dim) for t in valid_texts])
    X_test  = np.vstack([text_to_vec(t, embeddings, dim) for t in test_texts])

    return X_train, X_valid, X_test
