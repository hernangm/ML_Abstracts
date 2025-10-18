import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def train_model(model, X_train, y_train, cfg):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)

    # longitudes reales (no pads)
    pad = cfg.PAD_IDX if hasattr(cfg, "PAD_IDX") else 0
    lengths = (X_train != pad).sum(dim=1).to(torch.long)

    dataset = TensorDataset(X_train, lengths, y_train)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}")
        total_loss = 0.0
        for X, lengths, y in loop:
            X, lengths, y = X.to(cfg.DEVICE), lengths.to(cfg.DEVICE), y.to(cfg.DEVICE)
            optimizer.zero_grad()
            logits = model(X, lengths)
            loss = criterion(logits, y)   # y ∈ [0..C-1]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1}: Loss promedio {total_loss/len(loader):.4f}")
