import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class GRUClassifier(nn.Module):
    """
    GRU  text classification.

    Parameters
    vocab_size : int
        Vocabulary size.
    embed_dim : int
        Embedding dimension.
    hidden_dim : int
        GRU hidden units.
    num_classes : int
        Output classes.
    dropout : float
        Drop probability.
    num_layers : int
        GRU depth.
    pad_idx : int or None
        Padding token index.

    Key components
    embedding : token vectors.
    gru : recurrent encoder.
    packed : padded-aware sequence.
    h_n : final hidden state.
    fc : classification head.

    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout=0.5, num_layers=1, pad_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.embedding.weight)
        if pad_idx is not None:
            with torch.no_grad():
                self.embedding.weight[pad_idx].fill_(0)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        out = self.dropout(h_n[-1])
        return self.fc(out)
