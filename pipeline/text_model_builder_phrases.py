from text_preprocessor.tokenizer_phrases import build_vocab_phrases, text_to_tensor_phrases
from models.rnn_classifier import RNNClassifier


"""
    Prepare data and model components for an RNN classifier using phrase-aware tokenization.

    This function builds a vocabulary with bigram phrase detection, converts both
    training and test datasets into padded tensor representations, and initializes
    an RNN-based text classification model configured according to `cfg`.

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
        - DROPOUT: dropout rate for the model
        - NUM_LAYERS: number of RNN layers
        - PAD_IDX: index of the padding token
        - DEVICE: computation device ("cpu" or "cuda")

    Returns
    -------
    model : RNNClassifier
        Initialized RNN classification model.
    vocab : list of str
        List of tokens in the vocabulary including special tokens.
    stoi : dict
        Mapping from token string to integer index.
    X_train : torch.Tensor
        Tensor of token ID sequences for the training set.
    y_train : torch.Tensor
        Tensor of labels for the training set.
    X_test : torch.Tensor
        Tensor of token ID sequences for the test set.
    y_test : torch.Tensor
        Tensor of labels for the test set.
"""
### Paper link: https://arxiv.org/pdf/1310.4546


def prepare_rnn_phrases(df_train, df_test, cfg):

    print("Tokenizando con Phrases (Paper Mikolov 2013)...")
    vocab, stoi = build_vocab_phrases(df_train[cfg.TEXT_COL], min_freq=cfg.MIN_FREQ)

    X_train, y_train = text_to_tensor_phrases(df_train, stoi, cfg.MAX_LEN)
    X_test,  y_test  = text_to_tensor_phrases(df_test,  stoi, cfg.MAX_LEN)

    model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=cfg.EMBED_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.DROPOUT,
        num_layers=cfg.NUM_LAYERS,
        pad_idx=cfg.PAD_IDX
    ).to(cfg.DEVICE)

    return model, vocab, stoi, X_train, y_train, X_test, y_test
