import os
import sys

from run_model.runner import run_model

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config



def seleccionar_modelo():
    opciones = {
        "1": "rnn",
        "2": "lstm",
        "3": "gru",
        "4": "rnn_scheduler",
        "5": "rnn_phrases",
        "6": "bert_lora",
        "0": "salir"
    }

    while True:
        print("\nSeleccione el modelo a entrenar:")
        for k, v in opciones.items():
            print(f"{k}) {v}")
        opcion = input("\nOpción: ").strip()

        if opcion in opciones:
            return opciones[opcion]

        print("Opción inválida. Intente nuevamente.")


def main():

    while True:
        modelo = seleccionar_modelo()

        if modelo == "salir":
            print("Finalizando ejecución.")
            break

        print(f"\nModelo seleccionado: {modelo}\n")

        cfg = Config()
        cfg.MODEL_TYPE = modelo

        # === delegación total ===
        run_model(cfg)

        print("\nEntrenamiento finalizado. Volviendo al menú...\n")


if __name__ == "__main__":
    main()
