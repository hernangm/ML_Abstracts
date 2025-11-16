import numpy as np
from models.rnn_classifier import RNNClassifier

def softmax(v):
    e = np.exp(v - np.max(v))
    return e / np.sum(e)

def sample(model, seed_vec, n=100):
    h = np.zeros((model.hidden_size, 1))
    x = seed_vec
    output = []

    for _ in range(n):
        h = np.tanh(model.Wxh @ x + model.Whh @ h + model.bh)
        y = model.Why @ h + model.by
        p = softmax(y)

        ix = np.random.choice(range(model.Why.shape[0]), p=p.ravel())
        x = np.zeros_like(x)
        x[ix] = 1
        output.append(ix)

    return output


def entrenar_rnn(model, data, targets, lr=0.001):
    """
    Entrenamiento simple para RNN basado en udesa_ml2_rnn.py
    """
    for i in range(len(data)):
        inputs = data[i]
        targs = targets[i]

        outputs, last_h = model.forward(inputs)
        loss = model.loss(outputs, targs)

        # Aquí podrías agregar el backward del archivo original
        # si querés mantener exacta la implementación
        # backprop_rnn(model, outputs, inputs, targs, lr)

    return model


def evaluar_rnn(model, data, targets):
    total = 0
    correctos = 0

    for i in range(len(data)):
        outputs, _ = model.forward(data[i])
        _, y = outputs[-1]
        pred = np.argmax(y)
        if pred == targets[i][-1]:
            correctos += 1
        total += 1

    return correctos / total
