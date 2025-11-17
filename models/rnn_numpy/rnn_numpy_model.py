import numpy as np

class RNNNumPy:
    def __init__(self, vocab_size, hidden_size=100, seq_len=128):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # Parámetros del modelo
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        xs, hs, ys = {}, {}, {}

        for t in range(len(inputs)):
            xs[t] = inputs[t]

            h = np.tanh(self.Wxh @ xs[t] + self.Whh @ h + self.bh)
            hs[t] = h

            y = self.Why @ h + self.by
            ys[t] = y

        return ys, hs

    def loss(self, ys, targets):
        loss = 0
        for t in range(len(targets)):
            y = ys[t]
            p = np.exp(y - np.max(y))
            p = p / np.sum(p)
            loss += -np.log(p[targets[t], 0])
        return loss
