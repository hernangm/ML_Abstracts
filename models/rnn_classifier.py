import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout, num_layers=1, pad_idx=None):
        super().__init__()

        # Capa de embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # RNN “clásica” (no bidireccional)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_dim = hidden_dim  # como no es bidireccional, la salida es hidden_dim
        self.fc = nn.Linear(out_dim, num_classes)

        # Dropout en embeddings y capa final
        self.dropout = nn.Dropout(dropout)

        # Inicialización
        nn.init.xavier_uniform_(self.embedding.weight)
        if pad_idx is not None:
            with torch.no_grad():
                self.embedding.weight[pad_idx].fill_(0)

    def forward(self, x, lengths):
        """
        x: Tensor [batch, seq_len] con índices de vocabulario
        lengths: longitudes reales (sin PAD) de cada secuencia
        """

        # [batch, seq_len, embed_dim]
        emb = self.dropout(self.embedding(x))

        # Empaquetar para ignorar pads
        packed = pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # output: todas las salidas; h_n: último hidden de cada capa
        _, h_n = self.rnn(packed)

        # Nos quedamos con la última capa recurrente
        last_hidden = h_n[-1]          # [batch, hidden_dim]
        last_hidden = self.dropout(last_hidden)

        logits = self.fc(last_hidden)  # [batch, num_classes]
        return logits
