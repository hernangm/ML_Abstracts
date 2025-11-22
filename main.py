# main.py

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_loader.loader import load_dataset
from pipeline.text_model_builder import prepare_text_model
from train.trainer import train_model
from train.evaluator import evaluate_model
from utils.config import Config


def main():
    cfg = Config()

    print("Cargando dataset...")
    df_train, df_test, num_classes = load_dataset(
        cfg.DATA_PATH,
        text_col=cfg.TEXT_COL,
        label_col=cfg.LABEL_COL,
    )
    cfg.NUM_CLASSES = num_classes

    print(f"\n Preparando modelo PyTorch ({cfg.MODEL_TYPE.upper()})...")

    # ===========================================================
    #   Modelos especiales con comportamiento distinto:
    #   rnn_scheduler  → usa ReduceLROnPlateau
    # ===========================================================
    if cfg.MODEL_TYPE == "rnn_scheduler":
        (
            model,
            vocab,
            stoi,
            X_train,
            y_train,
            X_test,
            y_test,
            scheduler,
        ) = prepare_text_model(df_train, df_test, cfg)

        print("Entrenando modelo con Scheduler (ReduceLROnPlateau)...")
        train_model(model, X_train, y_train, cfg, scheduler=scheduler)

    # ===========================================================
    #   Modelos normales: rnn, lstm, gru, rnn_phrases
    # ===========================================================
    else:
        (
            model,
            vocab,
            stoi,
            X_train,
            y_train,
            X_test,
            y_test,
        ) = prepare_text_model(df_train, df_test, cfg)

        print("Entrenando modelo...")
        train_model(model, X_train, y_train, cfg)

    # ===========================================================
    #   Evaluación final
    # ===========================================================
    print("\nEvaluando modelo final...")
    evaluate_model(model, X_test, y_test, cfg)


if __name__ == "__main__":
    main()
