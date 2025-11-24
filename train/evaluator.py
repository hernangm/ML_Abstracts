import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np

"""
    Evaluate a sequence classification model on a test dataset.

    This function runs the model in evaluation mode, performs token-level
    masking to compute sequence lengths, generates predictions for each
    test sample, and computes overall classification accuracy.

    Supported model types include: 'rnn', 'lstm', 'gru', 'rnn_scheduler', and 'rnn_phrases'.

    Parameters
    ----------
    model : torch.nn.Module
        Trained sequence classification model.
    X_test : torch.Tensor
        Tensor containing padded token ID sequences of shape (N, T).
    y_test : torch.Tensor
        Tensor containing integer labels of shape (N,).
    cfg : object
        Configuration object with attributes described in @link utils/config.py
"""

def evaluate_model(model, X_test, y_test, cfg):
    model.eval()
    y_pred, y_true = [], []
    pad = cfg.PAD_IDX if hasattr(cfg, "PAD_IDX") else 0
    with torch.no_grad():
        for i in range(len(y_test)):
            sample = X_test[i].unsqueeze(0).to(cfg.DEVICE)     # [1, T]
            length = (sample != pad).sum(dim=1).to(torch.long) # [1]
            logits = model(sample, length)
            pred = torch.argmax(logits, dim=1).item()
            y_pred.append(pred)
            y_true.append(y_test[i].item())
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    precision, recall, f1 = compute_detailed_metrics(
        y_true,
        y_pred,
        cfg.NUM_CLASSES
    )
    print("\nMétricas por clase:")
    for c in range(cfg.NUM_CLASSES):
        print(f"Clase {c}:  Precision={precision[c]:.3f}  Recall={recall[c]:.3f}  F1={f1[c]:.3f}")


    plot_detailed_metrics(precision, recall, f1, cfg.NUM_CLASSES)

def compute_detailed_metrics(y_true, y_pred, num_classes):
    """
    Computes per-class precision
    recall
    F1 for a multiclass classifier.

    Parameters
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    num_classes : int
        Total class count.

    Key arguments
    precision : class-wise accuracy.
    recall : class-wise sensitivity.
    f1 : harmonic mean.
    labels : class indices.
    zero_division : safe fallback.

    Returns
    tuple
        precision, recall, f1
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0
    )
    return precision, recall, f1



def plot_detailed_metrics(precision, recall, f1, num_classes):
    """
    Plots per-class precision, recall, and F1 bar charts.

    Parameters
    precision : array-like
        Class-wise precision values.
    recall : array-like
        Class-wise recall values.
    f1 : array-like
        Class-wise F1 scores.
    num_classes : int
        Total number of classes.

    Key elements
    classes : class indices.
    width : bar spacing.
    x : bar positions.
    bar plots : metric bars.
    grid : y-axis grid.
    """


    classes = np.arange(num_classes)
    width = 0.25
    x = np.arange(len(classes))

    plt.figure(figsize=(12, 6))
    plt.title("Métricas de Clasificación Detalladas por Nota")

    plt.bar(x - width, precision, width=width, label="Precision")
    plt.bar(x, recall, width=width, label="Recall")
    plt.bar(x + width, f1, width=width, label="F1-Score")

    plt.xlabel("Nota (Clase Real)")
    plt.ylabel("Puntaje (0-1)")
    plt.xticks(classes, classes)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


