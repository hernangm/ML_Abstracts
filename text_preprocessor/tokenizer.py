import re
import torch
from collections import Counter

_word_re = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text):
    return _word_re.findall(text.lower())


"""
    Build a token vocabulary from a text corpus based on minimum frequency.

    This function tokenizes each document in the corpus, counts token
    frequencies, and constructs a vocabulary containing only tokens that
    occur at least `min_freq` times. Two special tokens, ``"<pad>"`` and
    ``"<unk>"``, are always included at the beginning of the vocabulary.

    Parameters
    ----------
    corpus : iterable of str
        A collection of raw text strings, such as a pandas Series or list.
    min_freq : int, optional (default=5)
        Minimum number of occurrences required for a token to be included
        in the vocabulary (excluding the special tokens).

    Returns
    -------
    vocab : list of str
        List of tokens in the vocabulary, beginning with ``"<pad>"`` and
        ``"<unk>"`` followed by all tokens meeting the frequency threshold.
    stoi : dict
        Dictionary mapping each token to its integer index within `vocab`.
"""


def build_vocab(corpus, min_freq=5):
    counter = Counter()
    for text in corpus:
        counter.update(tokenize(text))
    specials = ["<pad>", "<unk>"]
    vocab = specials + [tok for tok, c in counter.items() if c >= min_freq]
    stoi = {tok: i for i, tok in enumerate(vocab)}
    return vocab, stoi


"""
    Convert text and labels from a DataFrame into padded PyTorch tensors.

    This function tokenizes each text entry, maps tokens to integer IDs
    using the provided `stoi` vocabulary, truncates sequences longer than
    `max_len`, pads shorter sequences with the `<pad>` index, and returns
    tensors suitable for training PyTorch models.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the text and label columns.
    stoi : dict
        Mapping from token string to integer index, typically produced by
        `build_vocab`.
    max_len : int
        Maximum sequence length. Sequences longer than this are truncated,
        and shorter sequences are padded.
    text_col : str, optional (default="Body")
        Name of the column in `df` containing raw text strings.
    label_col : str, optional (default="label")
        Name of the column containing numeric labels.

    Returns
    -------
    X : torch.Tensor of shape (N, max_len)
        Tensor containing padded token ID sequences for all samples.
    y : torch.Tensor of shape (N,)
        Tensor containing label IDs.
"""


def text_to_tensor(df, stoi, max_len, text_col="Body", label_col="label"):
    pad_idx = stoi.get("<pad>", 0)
    unk_idx = stoi.get("<unk>", 1)

    def to_ids(text):
        ids = [stoi.get(tok, unk_idx) for tok in tokenize(text)]
        if len(ids) > max_len:
            ids = ids[:max_len]
        ids = torch.tensor(ids, dtype=torch.long)
        return torch.cat([ids, torch.full((max_len - len(ids),), pad_idx)])

    X = torch.stack([to_ids(t) for t in df[text_col]])
    y = torch.tensor(df[label_col].values, dtype=torch.long)
    return X, y
