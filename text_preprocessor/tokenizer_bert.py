from transformers import BertTokenizerFast
import torch

"""
    Tokenize text using BERT's tokenizer.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    tokens : list of str
        List of tokens produced by BERT tokenizer.
"""

def tokenize(text, tokenizer=None):
    if tokenizer is None:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return tokenizer.tokenize(text)


"""
    Build a token vocabulary using BERT's pretrained vocabulary.

    Parameters
    ----------
    None

    Returns
    -------
    vocab : list of str
        List of tokens in BERT's vocabulary.
    stoi : dict
        Dictionary mapping each token to its integer index within `vocab`.
"""

def build_vocab():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    vocab = list(tokenizer.vocab.keys())
    stoi = tokenizer.vocab
    return vocab, stoi


"""
    Convert a DataFrame of texts and labels to tensors using BERT's tokenizer.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing text and label columns.
    stoi : dict
        Token-to-index mapping from BERT's vocabulary.
    max_len : int
        Maximum sequence length.
    text_col : str, optional (default="Body")
        Name of the column containing text.
    label_col : str, optional (default="label")
        Name of the column containing labels.

    Returns
    -------
    X : torch.LongTensor
        Tensor of token indices, padded/truncated to `max_len`.
    y : torch.LongTensor
        Tensor of labels.
"""

def text_to_tensor(df, stoi, max_len, text_col="Body", label_col="label"):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    pad_idx = tokenizer.pad_token_id
    unk_idx = tokenizer.unk_token_id

    def to_ids(text):
        ids = tokenizer.encode(text, add_special_tokens=True, max_length=max_len, truncation=True)
        if len(ids) < max_len:
            ids += [pad_idx] * (max_len - len(ids))
        return torch.tensor(ids[:max_len], dtype=torch.long)

    X = torch.stack([to_ids(t) for t in df[text_col]])
    y = torch.tensor(df[label_col].values, dtype=torch.long)
    return X, y