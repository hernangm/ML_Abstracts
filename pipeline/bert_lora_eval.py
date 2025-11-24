import torch
from torch.utils.data import DataLoader, TensorDataset

def evaluate_bert_lora(model, X_ids, X_mask, y, cfg):
    """
       Evaluates BERT model fine-tuned  LoRA adapters.

       Parameters
       model : nn.Module
           BERT-LoRA classifier.
       X_ids : Tensor
           Input token IDs.
       X_mask : Tensor
           Attention masks.
       y : Tensor
           Ground-truth labels.
       cfg : Config
           Runtime configuration.

       Key arguments
       batch_size : eval batch size.
       device : compute device.
       logits : model scores.
       preds : predicted labels.
       accuracy : correct ratio.

       Returns
       float
           Classification accuracy over the dataset.
       """

    dataset = TensorDataset(X_ids, X_mask, y)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for ids, mask, labels in loader:
            ids = ids.to(cfg.DEVICE)
            mask = mask.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)

            outputs = model(input_ids=ids, attention_mask=mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"[BERT-LoRA] Accuracy: {acc:.4f}")
    return acc
