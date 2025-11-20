import os
import pandas as pd
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_submit(preds, name):
    df = pd.DataFrame({
        "Id": np.arange(1, len(preds)+1),
        "Prediction": preds
    })
    path = f"results/submission/submit_{name}.csv"
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
    print(f">>> Saved {path}")
    return path
