import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_bert_model():
    bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
    # Freeze BERT parameters for faster training
    for param in bert_model.parameters():
        param.requires_grad = False
    bert_model.to(device)
    return bert_model


# -------------------------------
# 3. Function to get BERT embeddings
# -------------------------------
def get_bert_embeddings(texts, model, max_length=128):
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