
from transformers import BertModel, BertTokenizerFast

def getModel(device):
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    for param in bert_model.parameters():
        param.requires_grad = False
    bert_model.to(device)
    return bert_model

def getTokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-uncased")
