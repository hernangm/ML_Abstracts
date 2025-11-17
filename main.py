import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_loader.loader import load_dataset
from pipeline.text_model_builder import prepare_text_model
from train.trainer import train_model
from train.evaluator import evaluate_model
from utils.config import Config

# === Solo para NumPy RNN ===
from text_preprocessor.tokenizer import build_vocab, tokenize


def construir_one_hot_sequences_numpy(df, stoi, max_len):
    import numpy as np
    vocab_size = len(stoi)

    pad = "<pad>"
    unk = "<unk>"
    pad_id = stoi.get(pad, 0)
    unk_id = stoi.get(unk, 1)

    X_sequences = []
    y_targets = []

    for _, row in df.iterrows():
        tokens = tokenize(row["Body"])
        ids = [stoi.get(tok, unk_id) for tok in tokens]

        # Corto
        if len(ids) > max_len:
            ids = ids[:max_len]

        # Guardo targets (índices)
        y_targets.append(ids)

        # Convierto a one-hot
        xs = []
        for idx in ids:
            v = np.zeros((vocab_size, 1))
            v[idx] = 1
            xs.append(v)

        # Padding
        while len(xs) < max_len:
            v = np.zeros((vocab_size, 1))
            v[pad_id] = 1
            xs.append(v)

        X_sequences.append(xs)

    return X_sequences, y_targets



def main():
    cfg = Config()

    print("Cargando dataset...")
    df_train, df_test, num_classes = load_dataset(
        cfg.DATA_PATH,
        text_col=cfg.TEXT_COL,
        label_col=cfg.LABEL_COL,
    )
    cfg.NUM_CLASSES = num_classes

    # ========================================================
    # 🚀 OPCIÓN 1 — Ejecutar la RNN NUMPY
    # ========================================================
    if cfg.MODEL_TYPE == "rnn_numpy":
        print("🧠 Ejecutando RNN NumPy ")

        # Construcción del vocabulario
        vocab, stoi = build_vocab(df_train[cfg.TEXT_COL], min_freq=cfg.MIN_FREQ)

        # One-hot encoding
        print("🔢 Construyendo secuencias one-hot...")
        X_train, y_train = construir_one_hot_sequences_numpy(df_train, stoi, cfg.MAX_LEN)
        X_test,  y_test  = construir_one_hot_sequences_numpy(df_test,  stoi, cfg.MAX_LEN)

        # Cargo la implementación NumPy
        from models.rnn_numpy.rnn_numpy_model import RNNNumPy
        from models.rnn_numpy.rnn_numpy_utils import entrenar_numpy, evaluar_numpy

        print("🧠 Inicializando RNN NumPy...")
        model = RNNNumPy(vocab_size=len(vocab), hidden_size=100)

        print("🏋 Entrenando RNN NumPy...")
        entrenar_numpy(model, X_train, y_train, lr=0.001, epochs=cfg.NUM_EPOCHS)

        print("📊 Evaluando RNN NumPy...")
        evaluar_numpy(model, X_test, y_test)

        return  # Importante: NO ejecutar flujo PyTorch



    # =========================================================
    # 🚀 OPCIÓN 2 — RNN / LSTM / GRU Pytorch
    # =========================================================
    print(f"Preparando modelo PyTorch ({cfg.MODEL_TYPE.upper()})...")
    model, vocab, stoi, X_train, y_train, X_test, y_test = prepare_text_model(
        df_train, df_test, cfg
    )

    print(f"Entrenando modelo {cfg.MODEL_TYPE.upper()}...")
    train_model(model, X_train, y_train, cfg)

    print(f"Evaluando modelo {cfg.MODEL_TYPE.upper()}...")
    evaluate_model(model, X_test, y_test, cfg)


if __name__ == "__main__":
    main()
