import torch
from text_preprocessor.tokenizer_bert import build_vocab, text_to_tensor
from transformers import BertForSequenceClassification

"""
    Prepare data, tokenizer, and model for a transformer-based classifier.

    This function loads BERT's vocabulary, converts training and test datasets
    into tensor representations, and initializes a BERT classification model.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training dataset containing the text and label columns.
    df_test : pandas.DataFrame
        Test dataset containing the text and label columns.
    cfg : object
        Configuration object with attributes:
        - TEXT_COL: name of the text column
        - MAX_LEN: maximum sequence length
        - NUM_CLASSES: number of output classes
        - DEVICE: computation device

    Returns
    -------
    model : BertForSequenceClassification
        Initialized BERT classification model.
    vocab : list of str
        BERT vocabulary list.
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
"""

def prepare_transformer_model(df_train, df_test, cfg):
    print("Loading BERT vocabulary...")
    vocab, stoi = build_vocab()

    print("Tokenizing and converting text to tensors...")
    X_train, y_train = text_to_tensor(df_train, stoi, cfg.MAX_LEN, text_col=cfg.TEXT_COL, label_col="label")
    X_test, y_test = text_to_tensor(df_test, stoi, cfg.MAX_LEN, text_col=cfg.TEXT_COL, label_col="label")

    print("🧠 Initializing BERT model for sequence classification...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=cfg.NUM_CLASSES
    ).to(cfg.DEVICE)

    return model, vocab, stoi, X_train, y_train, X_test, y_test