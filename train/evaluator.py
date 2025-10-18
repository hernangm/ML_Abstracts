import torch
from sklearn.metrics import accuracy_score

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
    print(f"✅ Accuracy: {acc*100:.2f}%")
