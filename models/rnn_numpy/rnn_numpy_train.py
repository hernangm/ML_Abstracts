import numpy as np
from .rnn_numpy_model import RNNNumPy

class RNNNumPyTrainer:
    def backward(self, model, xs, hs, ys, targets):
        dWxh = np.zeros_like(model.Wxh)
        dWhh = np.zeros_like(model.Whh)
        dWhy = np.zeros_like(model.Why)
        dbh = np.zeros_like(model.bh)
        dby = np.zeros_like(model.by)

        dh_next = np.zeros_like(hs[0])

        T = len(targets)

        for t in reversed(range(T)):
            y = ys[t]
            p = np.exp(y - np.max(y))
            p = p / np.sum(p)
            p[targets[t]] -= 1 # derivada de softmax + CE

            dWhy += p @ hs[t].T
            dby += p

            dh = (model.Why.T @ p) + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh
            dbh += dh_raw
            dWxh += dh_raw @ xs[t].T
            dWhh += dh_raw @ hs[t-1].T if t > 0 else 0
            dh_next = model.Whh.T @ dh_raw

        # Clipping
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return dWxh, dWhh, dWhy, dbh, dby

    def step(self, model, grads, lr):
        dWxh, dWhh, dWhy, dbh, dby = grads

        model.Wxh -= lr * dWxh
        model.Whh -= lr * dWhh
        model.Why -= lr * dWhy
        model.bh -= lr * dbh
        model.by -= lr * dby
