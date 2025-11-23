import torch
from text_preprocessor.tokenizer_phrases import build_vocab_phrases, text_to_tensor_phrases
from models.rnn_classifier import RNNClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau

"""
    Prepare data, vocabulary, model, optimizer, and learning-rate scheduler
    for an RNN classifier using phrase-aware tokenization.

    This function builds a vocabulary with bigram phrase detection, converts
    training and test datasets into tensor representations, initializes an
    RNN model, and configures an Adam optimizer together with a
    ReduceLROnPlateau scheduler.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training dataset containing the text and label columns.
    df_test : pandas.DataFrame
        Test dataset containing the text and label columns.
    cfg : object
        Configuration object with attributes:
        - TEXT_COL: name of the text column
        - MIN_FREQ: minimum token frequency for vocabulary inclusion
        - MAX_LEN: maximum sequence length
        - EMBED_DIM: embedding dimension
        - HIDDEN_DIM: RNN hidden size
        - NUM_CLASSES: number of output classes
        - DROPOUT: dropout rate
        - NUM_LAYERS: number of RNN layers
        - PAD_IDX: padding token index
        - DEVICE: computation device
        - LR: learning rate for the optimizer

    Returns
    -------
    model : RNNClassifier
        Initialized RNN classification model.
    vocab : list of str
        Vocabulary list including special tokens.
    stoi : dict
        Mapping from token string to integer index.
    X_train : torch.Tensor
        Tensor of token ID sequences for the training data.
    y_train : torch.Tensor
        Tensor of training labels.
    X_test : torch.Tensor
        Tensor of token ID sequences for the test data.
    y_test : torch.Tensor
        Tensor of test labels.
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler connected to the optimizer.
"""






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
