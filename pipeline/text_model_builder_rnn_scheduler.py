import torch
from text_preprocessor.tokenizer_phrases import build_vocab_phrases, text_to_tensor_phrases
from models.rnn_classifier import RNNClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau

def prepare_rnn_scheduler(df_train, df_test, cfg):

    print("Construyendo vocabulario con Phrases (opcional)...")
    vocab, stoi = build_vocab_phrases(df_train[cfg.TEXT_COL], min_freq=cfg.MIN_FREQ)

    X_train, y_train = text_to_tensor_phrases(df_train, stoi, cfg.MAX_LEN)
    X_test,  y_test  = text_to_tensor_phrases(df_test,  stoi, cfg.MAX_LEN)

    print("🧠 Inicializando modelo RNN con ReduceLROnPlateau...")
    model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=cfg.EMBED_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.DROPOUT,
        num_layers=cfg.NUM_LAYERS,
        pad_idx=cfg.PAD_IDX
    ).to(cfg.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    return model, vocab, stoi, X_train, y_train, X_test, y_test, scheduler
