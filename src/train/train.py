import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(model, train_loader, bert_model, device, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for input_ids, attention_mask, targets in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            with torch.no_grad():  # get BERT embeddings
                outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token

            optimizer.zero_grad()
            predictions = model(embeddings)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {total_loss / len(train_loader):.4f}")