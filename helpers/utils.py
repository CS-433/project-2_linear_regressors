import os
import pandas as pd
import numpy as np
import random
import torch

def ensure_dir(path):
    '''
    Ensure that a directory exists; if not, create it.
    '''
    os.makedirs(path, exist_ok=True)

def save_submit(preds, name):
    '''
    Save predictions in submission format
    Args:
        preds: list of predictions
        name: name of the submission file
    Returns:
        path to the saved submission file
    '''
    df = pd.DataFrame({
        "Id": np.arange(1, len(preds)+1),
        "Prediction": preds
    })
    df['Prediction'] = df['Prediction'].apply(lambda x: -1 if x == 0 else 1)
    path = f"results/submission/submit_{name}.csv"
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
    print(f">>> Saved {path}")
    return path

def set_global_seed(seed=42):
    '''
    Set the global random seed for reproducibility
    Args:
        seed: the seed value to set
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # MPS determinism
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # Ensure deterministic ops
    torch.use_deterministic_algorithms(True, warn_only=True)

    os.environ["PYTHONHASHSEED"] = str(seed)
