import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM para CLASIFICACIÓN de texto.

    - Mantiene la misma interfaz que el resto del proyecto:
      * __init__(vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.5, num_layers=1, pad_idx=None)
      * forward(x, lengths) -> logits [batch, num_classes]

    - Usa 'lengths' para ignorar correctamente el padding:
      * Elige el último timestep REAL de cada secuencia (no el <pad>).
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_classes,
        dropout=0.5,
        num_layers=1,
        pad_idx=None,
    ):
        super().__init__()

        # ===== Embedding =====
        # padding_idx permite que el token <pad> no aporte información
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # ===== LSTM =====
        # batch_first=True → x tiene forma [batch, seq_len, embed_dim]
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ===== Capa final de clasificación =====
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Dropout en embeddings y en la salida antes de la FC
        self.dropout = nn.Dropout(dropout)

        # ===== Inicialización razonable =====
        # Embeddings Xavier y padding en cero
        nn.init.xavier_uniform_(self.embedding.weight)
        if pad_idx is not None:
            with torch.no_grad():
                self.embedding.weight[pad_idx].fill_(0)

        # Inicialización ortogonal de las matrices recurrentes (opcional pero prolijo)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)

    def forward(self, x, lengths):
        """
        x:        Tensor [batch, seq_len] con índices de vocabulario
        lengths:  Tensor [batch] con la cantidad de tokens NO-PAD de cada secuencia

        Devuelve:
        logits:   Tensor [batch, num_classes]
        """

        # 1) Embedding + dropout
        #    [B, T] → [B, T, E]
        emb = self.dropout(self.embedding(x))

        # 2) Pasamos todo por la LSTM
        #    lstm_out: [B, T, H]
        #    h_n, c_n: [num_layers, B, H] (no los usamos directamente)
        lstm_out, (h_n, c_n) = self.lstm(emb)

        # 3) Seleccionar el "último timestep REAL" de cada secuencia
        #    lengths indica cuántos tokens reales tiene cada ejemplo (sin pads).
        #    El índice del último real es lengths - 1.
        #
        #    Ejemplo:
        #      seq:    [w1 w2 w3 <pad> <pad>]
        #      length: 3  → índice último real = 2
        #
        #    lstm_out: [B, T, H]
        #      → last_hidden: [B, H]

        # Aseguramos que no haya longitudes cero (por seguridad)
        lengths = lengths.clamp(min=1)

        batch_size = x.size(0)
        device = x.device

        # Índices [0, 1, 2, ..., B-1]
        batch_indices = torch.arange(batch_size, device=device)

        # Índice del último timestep real para cada secuencia
        time_indices = lengths - 1  # [B]

        # Tomamos el vector correspondiente a (batch, time_indices[i]) para todos
        last_hidden = lstm_out[batch_indices, time_indices, :]  # [B, H]

        # 4) Dropout + capa final
        out = self.dropout(last_hidden)
        logits = self.fc(out)  # [B, num_classes]

        return logits
