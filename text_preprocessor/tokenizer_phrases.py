import torch
import re
from collections import Counter
from gensim.models.phrases import Phrases

_word_re = re.compile(r"\b\w+\b", re.UNICODE)

def tokenize(text):
    return _word_re.findall(text.lower())

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
