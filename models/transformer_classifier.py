import torch
import torch.nn as nn
from transformers import BertModel

class TransformerRegressor(nn.Module):
    """
    Regresor basado en transformers para texto.

    Esta clase implementa un modelo de regresión usando BERT como backbone.
    El modelo toma secuencias de texto, las procesa con BERT y aplica una capa lineal
    para obtener una salida continua (regresión).

    - Interfaz:
      * __init__(num_classes=1, pretrained_model="bert-base-uncased", dropout=0.1)
        Inicializa el modelo, cargando BERT y definiendo la capa de regresión.
      * forward(x, attention_mask=None) -> output [batch, num_classes]
        Realiza la pasada hacia adelante, obteniendo la representación del texto
        y aplicando la capa de regresión.

    - Utiliza BERT como base y una cabeza de regresión (capa lineal).
    """

    def __init__(self, num_classes=1, pretrained_model="bert-base-uncased", dropout=0.1):
        # Inicializa el modelo BERT y la capa de regresión
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, attention_mask=None):
        # Realiza la pasada hacia adelante:
        # 1. Obtiene la salida de BERT (pooler_output es la representación global del texto).
        # 2. Aplica dropout para regularización.
        # 3. Aplica la capa lineal para obtener la predicción de regresión.
        outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [batch, hidden_size]
        out = self.dropout(pooled)
        return self.fc(out)