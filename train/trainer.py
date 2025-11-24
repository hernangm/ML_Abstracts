import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


"""
Entrena un modelo PyTorch para clasificación de texto.

Parameters
----------
model : torch.nn.Module
    Modelo recurrente (RNN, LSTM, GRU).
X_train : Tensor
    Tensor [N, T] con índices del vocabulario.
y_train : Tensor
    Tensor [N] con labels enteros.
cfg : Config
    Configuración del proyecto (lr, batch, etc.).
scheduler : ReduceLROnPlateau | None
    Scheduler opcional.
"""


def train_model(
    model: torch.nn.Module,
    X_train: Tensor,
    y_train: Tensor,
    cfg,
    scheduler=None
):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)

    pad_id = cfg.PAD_IDX
    mask: Tensor = (X_train != pad_id)
    lengths: Tensor = mask.sum(dim=1).to(torch.long)

    dataset = TensorDataset(X_train, mask, lengths, y_train)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS}")

        for X, attention_mask, seq_lengths, y in loop:
            X = X.to(cfg.DEVICE)
            attention_mask = attention_mask.to(cfg.DEVICE)
            seq_lengths = seq_lengths.to(cfg.DEVICE)
            y = y.to(cfg.DEVICE)

            optimizer.zero_grad()

            if cfg.MODEL_TYPE == "transformer":
                logits = model(X, attention_mask=attention_mask).logits
            else:
                logits = model(X, seq_lengths)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}: Loss promedio {avg_loss:.4f}")

        # === Scheduler opcional ===
        if scheduler is not None:
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  → Scheduler: LR actualizado a {current_lr:.8f}")
