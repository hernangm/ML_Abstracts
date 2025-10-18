from text_preprocessor.tokenizer import build_vocab, text_to_tensor
from models.gru_classifier import GRUClassifier

def prepare_gru_model(df_train, df_test, cfg):
    print("🔠 Construyendo vocabulario y tokenizando...")
    vocab, stoi = build_vocab(df_train[cfg.TEXT_COL], min_freq=cfg.MIN_FREQ)
    X_train, y_train = text_to_tensor(df_train, stoi, cfg.MAX_LEN, label_col="label")
    X_test,  y_test  = text_to_tensor(df_test,  stoi, cfg.MAX_LEN, label_col="label")

    print("🧠 Inicializando modelo GRU...")
    model = GRUClassifier(
        vocab_size=len(vocab),
        embed_dim=cfg.EMBED_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.DROPOUT,
        num_layers=cfg.NUM_LAYERS,
        pad_idx=cfg.PAD_IDX
    ).to(cfg.DEVICE)

    return model, vocab, stoi, X_train, y_train, X_test, y_test
