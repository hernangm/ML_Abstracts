import torch
import re
from collections import Counter
from gensim.models.phrases import Phrases

_word_re = re.compile(r"\b\w+\b", re.UNICODE)

def tokenize(text):
    return _word_re.findall(text.lower())

"""
    Build a vocabulary with bigram phrase detection from a corpus of text.

    This function tokenizes each document, learns bigram collocations using
    `gensim.models.Phrases`, applies the frozen bigram model to produce
    phrase-aware token sequences, counts token frequencies, and finally
    constructs a vocabulary filtered by `min_freq`.

    Parameters
    ----------
    corpus : A series of raw text strings.
    min_freq : Minimum number of occurrences required for a token to be included
               in the vocabulary (excluding special tokens).

    Returns
    -------
    vocab : list of str
        List of tokens in the vocabulary, including special tokens
        ``"<pad>"`` and ``"<unk>"`` at the beginning.
    stoi : dict
        A mapping from token to integer index. Indices align with
        positions in ``vocab``.
"""

def build_vocab_phrases(corpus, min_freq=5):
    sentences = [tokenize(t) for t in corpus]

    print("Construyendo bigramas con Phrases...")
    phrases = Phrases(sentences, min_count=1, threshold=4)
    bigram = phrases.freeze()

    bigram_sentences = [bigram[s] for s in sentences]

    counter = Counter()
    for s in bigram_sentences:
        counter.update(s)

    specials = ["<pad>", "<unk>"]
    vocab = specials + [tok for tok, c in counter.items() if c >= min_freq]

    stoi = {tok: i for i, tok in enumerate(vocab)}

    return vocab, stoi


"""
    Convert a text dataset into padded integer tensors using a phrase-aware vocabulary.

    This function tokenizes text, converts tokens to integer IDs using `stoi`,
    truncates sequences to `max_len`, pads them to equal length, and returns
    tensors suitable for training PyTorch models.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the text and label columns.
    stoi : dict
        Mapping from token (string) to integer index, typically produced by
        `build_vocab_phrases`.
    max_len : int
        Maximum allowed sequence length. Sequences longer than this are
        truncated; shorter sequences are padded with the `<pad>` index.
    text_col : str, optional (default="Body")
        Name of the column in `df` containing raw text.
    label_col : str, optional (default="label")
        Name of the column in `df` containing label values.

    Returns
    -------
    X : torch.Tensor of shape (N, max_len)
        Tensor containing padded token ID sequences.
    y : torch.Tensor of shape (N,)
        Tensor containing integer labels.
"""

def text_to_tensor_phrases(df, stoi, max_len, text_col="Body", label_col="label"):
    pad_idx = stoi.get("<pad>", 0)
    unk_idx = stoi.get("<unk>", 1)

    def to_ids(text):
        tokens = tokenize(text)
        ids = [stoi.get(t, unk_idx) for t in tokens]
        if len(ids) > max_len:
            ids = ids[:max_len]
        ids = torch.tensor(ids, dtype=torch.long)
        return torch.cat([ids, torch.full((max_len - len(ids),), pad_idx)])

    X = torch.stack([to_ids(t) for t in df[text_col]])
    y = torch.tensor(df[label_col].values, dtype=torch.long)

    return X, y
