# Sara
'''
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_sklearn(model, X_train, y_train, X_valid=None, y_valid=None):
    # Train
    model.train(X_train, y_train)

    # Eval si valid set fourni
    if X_valid is not None and y_valid is not None:
        preds = model.predict(X_valid)
        acc = accuracy_score(y_valid, preds)
        prec = precision_score(y_valid, preds, zero_division=0)
        rec = recall_score(y_valid, preds, zero_division=0)
        f1 = f1_score(y_valid, preds, zero_division=0)

        print(f"[EVAL] acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")
    
    return model
'''