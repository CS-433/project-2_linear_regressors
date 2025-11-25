import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from helpers.utils import ensure_dir
import numpy as np

def plot_confusion_matrix(preds, true_labels, name, train_size):
    if train_size == 2_250_000:
        train_size = "full"
    ensure_dir("results/individual_plots")
    cm = confusion_matrix(true_labels, preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True, fmt="d", cmap="Blues",
        xticklabels=["neg","pos"], yticklabels=["neg","pos"]
    )
    plt.title(f"Confusion matrix for {name} with {train_size} tweets")
    path = f"results/individual_plots/{name}/cm_{name}_{train_size}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def plot_training_curves(logger, model_name, train_size):
    if train_size == 2_250_000:
        train_size = "full"
    ensure_dir("results/individual_plots")

    # train loss
    steps, losses = zip(*logger.train_loss)
    plt.plot(steps, losses)
    plt.title(f"Train Loss – {model_name} – {train_size}")
    path = f"results/individual_plots/{model_name}/trainloss_{model_name}_{train_size}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    # eval loss
    if logger.eval_loss:
        steps, losses = zip(*logger.eval_loss)
        plt.plot(steps, losses)
        plt.title(f"Eval Loss – {model_name} – {train_size}")
        path2 = f"results/individual_plots/{model_name}/evalloss_{model_name}_{train_size}.png"
        os.makedirs(os.path.dirname(path2), exist_ok=True)
        plt.savefig(path2, dpi=200, bbox_inches="tight")
        plt.close()

    # eval accuracy
    if logger.eval_acc:
        steps, accs = zip(*logger.eval_acc)
        plt.plot(steps, accs)
        plt.title(f"Eval Accuracy – {model_name} – {train_size}")
        path3 = f"results/individual_plots/{model_name}/evalacc_{model_name}_{train_size}.png"
        os.makedirs(os.path.dirname(path3), exist_ok=True)
        plt.savefig(path3, dpi=200, bbox_inches="tight")
        plt.close()


def save_metrics(name, acc, prec, rec, f1, path="results/comparison_plots/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    entry = {
        "name": name,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1
    }

    # Load previous results if file exists
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = []

    # remove previous entries with the same name
    data = [d for d in data if d["name"] != name]

    data.append(entry)

    # Save back
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[METRICS] Saved metrics for {name}")


def plot_comparison(models=None, labels=None,
                    metrics_path="results/comparison_plots/metrics.json",
                    output_name="comparison"):
    """
    Compare selected models using the saved metrics.json.
    
    Args:
        models (list[str]) : list of model names saved in metrics.json
        labels (list[str]) : list of display names (same length as models)
        metrics_path (str) : path to the stored metrics.json file
        output_name (str)  : name of the saved PNG output
    """

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")

    # Load stored metrics
    with open(metrics_path, "r") as f:
        all_metrics = json.load(f)

    # If no model list is provided → use ALL saved models
    if models is None:
        models = [m["name"] for m in all_metrics]

    # If labels are not provided = use model names
    if labels is None:
        labels = models

    if len(models) != len(labels):
        raise ValueError("models and labels must have the same length")

    # Filter metrics to only those requested
    selected = [m for m in all_metrics if m["name"] in models]

    if len(selected) == 0:
        raise ValueError("None of the requested models were found in metrics.json")

    # Extract scores
    model_names = labels
    accuracies  = [m["acc"]  for m in selected]
    precisions  = [m["prec"] for m in selected]
    recalls     = [m["rec"]  for m in selected]
    f1_scores   = [m["f1"]   for m in selected]

    # Plot
    x = np.arange(len(model_names))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5*width, accuracies,  width, label="Accuracy")
    plt.bar(x - 0.5*width, precisions,  width, label="Precision")
    plt.bar(x + 0.5*width, recalls,     width, label="Recall")
    plt.bar(x + 1.5*width, f1_scores,   width, label="F1 Score")

    plt.xticks(x, model_names, rotation=20, ha="right")
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()

    ensure_dir("results/comparison_plots")

    # Filename includes only the chosen models
    short_name = "_".join([str(m).replace("/", "-") for m in models]) # jai changé ca
    png_path = f"results/comparison_plots/{output_name}_{short_name}.png"

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"[PLOT] Saved comparison plot → {png_path}")
