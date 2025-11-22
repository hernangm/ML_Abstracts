import torch
import os

class Config:
    # Directorio base
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Ruta completa al dataset
    DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "processed", "Abstracts.xlsx")

    # Nombre de la columna
    TEXT_COL = "Body"

    # Nombre de la columna puntajes
    LABEL_COL = "Scores"

    # Tipo de modelo
    # Opciones: 'rnn', 'lstm', 'gru', 'rnn_scheduler', 'rnn_phrases'
    MODEL_TYPE = "lstm"

    # Semilla
    SEED = 42

    # Longitud máxima
    MAX_LEN = 128

    # Tamaño del batch
    BATCH_SIZE = 32

    # Numero total de épocas
    EPOCHS = 10

    # Dimensión del embedding
    EMBED_DIM = 128

    # Tamaño de la capa oculta  RNN/LSTM/GRU
    HIDDEN_DIM = 64

    # Cantidad de capas
    NUM_LAYERS = 1

    # Probabilidad de dropout
    DROPOUT = 0.5

    # Learning rate del optimizador Adam
    LR = 1e-3

    # Dispositivo GPU - CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Frecuencia mínima
    MIN_FREQ = 1

    # Cantidad de clases
    NUM_CLASSES = 5

    # padding
    PAD_IDX = 0

    # EPOCHS
    NUM_EPOCHS = EPOCHS

    # ===== LoRA (Low-Rank Adaptation) =====
    #  LoRA
    USE_LORA = False

    # factorización de LoRA
    LORA_R = 16

    # Parámetro de escalado de LoRA
    LORA_ALPHA = 32

    # Dropout aplicado
    LORA_DROPOUT = 0.07

    # Lista de nombres
    LORA_TARGET_MODULES = ["weight_hh_l0", "weight_ih_l0"]

