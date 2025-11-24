import torch
from sklearn.metrics import accuracy_score, f1_score
import os
import pandas as pd
from datetime import datetime


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
    """
    Evalúa el modelo y guarda los resultados en un archivo CSV en la carpeta 'results'.
    El archivo incluye los valores objetivo, predichos y la diferencia absoluta.
    """
    model.eval()
    y_pred, y_true = [], []
    pad = cfg.PAD_IDX if hasattr(cfg, "PAD_IDX") else 0
    with torch.no_grad():
        for i in range(len(y_test)):
            sample = X_test[i].unsqueeze(0).to(cfg.DEVICE)     # [1, T]
            length = (sample != pad).sum(dim=1).to(torch.long) # [1]
            if cfg.MODEL_TYPE == "transformer":
                mask = (sample != pad).to(cfg.DEVICE)
                logits = model(sample, attention_mask=mask).logits
            else:
                logits = model(sample, length)
            pred = torch.argmax(logits, dim=1).item()
            y_pred.append(pred)
            y_true.append(y_test[i].item())
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    abs_diff = [abs(t - p) for t, p in zip(y_true, y_pred)]
    avg_abs_diff = sum(abs_diff) / len(abs_diff) if abs_diff else 0
    print(f"Promedio de la diferencia absoluta: {avg_abs_diff:.4f}")
    # Calcular F1 score (macro para clasificación multiclase)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"F1 score (macro): {f1:.4f}")

    # Guardar resumen en archivo aparte
    os.makedirs("results", exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"results/{cfg.MODEL_TYPE}_{now}_summary.csv"
    summary_df = pd.DataFrame({
        "accuracy": [acc],
        "avg_abs_diff": [avg_abs_diff],
        "f1_score_macro": [f1]
    })
    summary_df.to_csv(summary_filename, index=False)
    print(f"Resumen guardado en: {summary_filename}")

    # Guardar resultados en CSV
    filename = f"results/{cfg.MODEL_TYPE}_{now}.csv"
    data = {
        "target": y_true,
        "prediction": y_pred,
        "abs_diff": abs_diff
    }
    df = pd.DataFrame(data)
    df = df[["target", "prediction", "abs_diff"]]
    df.to_csv(filename, index=False)
    print(f"Resultados guardados en: {filename}")
