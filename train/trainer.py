# train/trainer.py

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


def train_model(
    model,
    X_train,
    y_train,
    cfg,
    scheduler=None,
    X_test=None,
    y_test=None
):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)

    pad_id = cfg.PAD_IDX
    lengths_train = (X_train != pad_id).sum(dim=1).to(torch.long)

    dataset_train = TensorDataset(X_train, lengths_train, y_train)
    loader_train = DataLoader(dataset_train, batch_size=cfg.BATCH_SIZE, shuffle=True)

    # === MÉTRICAS POR ÉPOCA ===
    history = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": [],
        "test_f1": [],
    }

    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        loop = tqdm(loader_train, desc=f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}")
        for Xb, Lb, yb in loop:
            Xb = Xb.to(cfg.DEVICE)
            Lb = Lb.to(cfg.DEVICE)
            yb = yb.to(cfg.DEVICE)

            optimizer.zero_grad()
            logits = model(Xb, Lb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(loader_train)
        history["train_loss"].append(avg_train_loss)

        # === Evaluación por época ===
        if X_test is not None:
            test_loss, test_acc, test_f1 = _evaluate_epoch(model, X_test, y_test, cfg)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)
            history["test_f1"].append(test_f1)

            print(f"  → Test Loss: {test_loss:.4f} | Acc: {test_acc:.3f} | F1: {test_f1:.3f}")

            # Scheduler opcional
            if scheduler is not None:
                scheduler.step(test_loss)
                print(f"  → Scheduler LR = {optimizer.param_groups[0]['lr']:.8f}")

    # === Dibujar gráficos ===
    _plot_training_curves(history)

    return history



def _evaluate_epoch(model, X_test, y_test, cfg):
    """
    Devuelve: test_loss, accuracy, f1
    """
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    pad = cfg.PAD_IDX
    lengths = (X_test != pad).sum(dim=1).to(torch.long)

    dataset = TensorDataset(X_test, lengths, y_test)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE)

    total_loss = 0.0
    y_pred, y_true = [], []

    with torch.no_grad():
        for Xb, Lb, yb in loader:
            Xb = Xb.to(cfg.DEVICE)
            Lb = Lb.to(cfg.DEVICE)
            yb = yb.to(cfg.DEVICE)

            logits = model(Xb, Lb)
            loss = criterion(logits, yb)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(yb.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return avg_loss, acc, f1



def _plot_training_curves(history):
    """
    Genera gráficos de:
    - Train Loss
    - Test Loss
    - Test Accuracy
    - Test F1
    """

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 10))

    # === Train vs Test Loss ===
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.title("Loss por época")
    plt.legend()

    # === Accuracy ===
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["test_acc"], label="Test Accuracy")
    plt.title("Accuracy por época")

    # === F1 ===
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["test_f1"], label="Test F1 Score")
    plt.title("F1 Score por época")

    plt.tight_layout()
    plt.show()
