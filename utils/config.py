import torch
import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "processed", "Abstracts.xlsx")

    TEXT_COL = "Body"
    LABEL_COL = "Scores"
    MODEL_TYPE = "rnn"   # opciones: 'rnn', 'lstm', 'gru'

    SEED = 42
    MAX_LEN = 128
    BATCH_SIZE = 32
    EPOCHS = 5
    EMBED_DIM = 128
    HIDDEN_DIM = 64
    NUM_LAYERS = 1
    DROPOUT = 0.5
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIN_FREQ = 1
    NUM_CLASSES = 5
    PAD_IDX = 0
    NUM_EPOCHS = EPOCHS

    # === LoRA ===
    USE_LORA = True
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.07
    LORA_TARGET_MODULES = ["weight_hh_l0", "weight_ih_l0"]  # capas recurrentes
