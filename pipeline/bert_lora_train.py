import torch
from torch.utils.data import DataLoader, TensorDataset

def train_bert_lora(model, X_ids, X_mask, y, cfg):
    def train_bert_lora(model, X_ids, X_mask, y, cfg):
        """
        Fine-tunes a BERT model augmented with LoRA adapters.

        Parameters
        model : nn.Module
            BERT LoRA model.
        X_ids : Tensor
            Input token IDs.
        X_mask : Tensor
            Attention masks.
        y : Tensor
            Class labels.
        cfg : Config
            Runtime configuration.

        Key arguments
        batch_size : training batch size.
        lr : learning rate.
        epochs : total iterations.
        device : compute device.
        criterion : loss function.
        optimizer : weight update rule.

        Returns
        None

        """

    dataset = TensorDataset(X_ids, X_mask, y)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(cfg.EPOCHS):
        total_loss = 0.0

        for ids, mask, labels in loader:
            ids = ids.to(cfg.DEVICE)
            mask = mask.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)

            optimizer.zero_grad()

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[BERT-LoRA] Epoch {epoch+1}/{cfg.EPOCHS} - Loss: {total_loss/len(loader):.4f}")
