import torch
from transformers import AutoTokenizer, AutoModel


def get_bert_model(device):
    bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
    for param in bert_model.parameters():
        param.requires_grad = False
    bert_model.to(device)
    return bert_model


def get_bert_embeddings(texts, model, device, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer(
        texts.tolist(),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Take [CLS] token embeddings
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu()