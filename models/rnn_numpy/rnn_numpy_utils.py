import numpy as np
from .rnn_numpy_model import RNNNumPy
from .rnn_numpy_train import RNNNumPyTrainer

trainer = RNNNumPyTrainer()

def entrenar_numpy(model, X, y, lr=0.001, epochs=3):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            xs = X[i]      # lista de one-hot
            ts = y[i]      # targets

            ys, hs = model.forward(xs)
            loss = model.loss(ys, ts)
            total_loss += loss

            grads = trainer.backward(model, xs, hs, ys, ts)
            trainer.step(model, grads, lr)

        print(f"Epoch {epoch+1}: Loss {total_loss/len(X):.4f}")

    return model

def evaluar_numpy(model, X, y):
    correct = 0
    total = len(X)

    for i in range(total):
        ys, _ = model.forward(X[i])
        last = ys[len(ys)-1]
        pred = np.argmax(last)
        if pred == y[i][-1]:
            correct += 1

    acc = correct / total
    print(f"Accuracy NumPy: {acc*100:.2f}%")
    return acc
