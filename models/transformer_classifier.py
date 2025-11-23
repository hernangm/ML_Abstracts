import torch
import torch.nn as nn
from transformers import BertModel

class TransformerRegressor(nn.Module):
    """
    Transformer-based regressor for text.

    - Interface:
      * __init__(num_classes=1, pretrained_model="bert-base-uncased", dropout=0.1)
      * forward(x, attention_mask=None) -> output [batch, num_classes]

    - Uses BERT as backbone and a regression head.
    """

    def __init__(self, num_classes=1, pretrained_model="bert-base-uncased", dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, attention_mask=None):
        outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [batch, hidden_size]
        out = self.dropout(pooled)
        return self.fc(out)