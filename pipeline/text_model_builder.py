from text_preprocessor.tokenizer import build_vocab, text_to_tensor

def prepare_text_model(df_train, df_test, cfg):
    print("Construyendo vocabulario y tokenizando...")
    vocab, stoi = build_vocab(df_train[cfg.TEXT_COL], min_freq=cfg.MIN_FREQ)
    X_train, y_train = text_to_tensor(df_train, stoi, cfg.MAX_LEN, label_col="label")
    X_test, y_test = text_to_tensor(df_test, stoi, cfg.MAX_LEN, label_col="label")

    print(f" Inicializando modelo ({cfg.MODEL_TYPE})...")

    if cfg.MODEL_TYPE == "rnn":
        from models.rnn_classifier import RNNClassifier as Model
    elif cfg.MODEL_TYPE == "lstm":
        from models.lstm_classifier import LSTMClassifier as Model
    elif cfg.MODEL_TYPE == "gru":
        from models.gru_classifier import GRUClassifier as Model
    elif cfg.MODEL_TYPE == "rnn_scheduler":
        from pipeline.text_model_builder_rnn_scheduler import prepare_rnn_scheduler
        model, vocab, stoi, X_train, y_train, X_test, y_test, scheduler = \
            prepare_rnn_scheduler(df_train, df_test, cfg)
        return model, vocab, stoi, X_train, y_train, X_test, y_test, scheduler
    elif cfg.MODEL_TYPE == "rnn_phrases":
        from pipeline.text_model_builder_phrases import prepare_rnn_phrases
        return prepare_rnn_phrases(df_train, df_test, cfg)
    else:
        raise ValueError(f"Modelo no soportado: {cfg.MODEL_TYPE}")

    model = Model(
        vocab_size=len(vocab),
        embed_dim=cfg.EMBED_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.DROPOUT,
        num_layers=cfg.NUM_LAYERS,
        pad_idx=cfg.PAD_IDX
    ).to(cfg.DEVICE)

    # === Aplicar LoRA si está activado ===
    if getattr(cfg, "USE_LORA", False):
        from utils.lora_utils import apply_lora
        print("🔧 Aplicando LoRA...")
        model = apply_lora(model, cfg)

    return model, vocab, stoi, X_train, y_train, X_test, y_test
