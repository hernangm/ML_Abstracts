# main.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_loader.loader import load_dataset
from pipeline.text_model_builder import prepare_text_model
from train.trainer import train_model
from train.evaluator import evaluate_model
from utils.config import Config


def seleccionar_modelo():
    opciones = {
        "1": "rnn",
        "2": "lstm",
        "3": "gru",
        "4": "rnn_scheduler",
        "5": "rnn_phrases",
        "6": "transformer",
        "0": "salir"
    }

    while True:
        print("\nSeleccione el modelo a entrenar:")
        print("1) rnn")
        print("2) lstm")
        print("3) gru")
        print("4) rnn_scheduler")
        print("5) rnn_phrases")
        print("6) transformer")
        print("0) salir\n")

        opcion = input("Opción: ").strip()

        if opcion in opciones:
            return opciones[opcion]

        print("Opción inválida. Intente nuevamente.")


def ejecutar_entrenamiento(model_type):
    cfg = Config()
    cfg.MODEL_TYPE = model_type

    print("\nCargando dataset...")
    df_train, df_test, num_classes = load_dataset(
        cfg.DATA_PATH,
        text_col=cfg.TEXT_COL,
        label_col=cfg.LABEL_COL,
    )
    cfg.NUM_CLASSES = num_classes

    print(f"\nPreparando modelo ({cfg.MODEL_TYPE})...")

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

        print("Entrenando modelo con ReduceLROnPlateau...")
        train_model(model, X_train, y_train, cfg, scheduler=scheduler)

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

    print("\nEvaluando modelo final...")
    evaluate_model(model, X_test, y_test, cfg)


def main():

    while True:
        modelo = seleccionar_modelo()

        if modelo == "salir":
            print("Finalizando ejecución.")
            break

        print(f"\nModelo seleccionado: {modelo}")
        ejecutar_entrenamiento(modelo)
        print("\nEntrenamiento finalizado. Volviendo al menú...\n")


if __name__ == "__main__":
    main()
