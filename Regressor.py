import torch
import torch.nn as nn
import torch.optim as optim


class TextRegressor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train(model, train_loader, bert_model, device, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0

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

        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {total_loss / len(train_loader):.4f}")


