from transformers import BertTokenizerFast
import torch

"""
    Tokeniza texto usando el tokenizador de BERT.

    Esta función recibe un texto en bruto y lo convierte en una lista de tokens
    utilizando el tokenizador preentrenado de BERT. Si no se proporciona un tokenizador,
    se carga automáticamente el modelo "bert-base-uncased".

    Parámetros
    ----------
    text : str
        Texto de entrada sin procesar.
    tokenizer : BertTokenizerFast, opcional
        Instancia del tokenizador de BERT. Si es None, se carga automáticamente.

    Retorna
    -------
    tokens : list of str
        Lista de tokens generados por el tokenizador de BERT.
"""

def tokenize(text, tokenizer=None):
    if tokenizer is None:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return tokenizer.tokenize(text)


"""
    Construye el vocabulario de tokens usando el vocabulario preentrenado de BERT.

    Esta función carga el vocabulario del modelo "bert-base-uncased" y retorna
    tanto la lista de tokens como el diccionario que mapea cada token a su índice.

    Parámetros
    ----------
    Ninguno

    Retorna
    -------
    vocab : list of str
        Lista de todos los tokens en el vocabulario de BERT.
    stoi : dict
        Diccionario que mapea cada token a su índice entero dentro del vocabulario.
"""

def build_vocab():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    vocab = list(tokenizer.vocab.keys())
    stoi = tokenizer.vocab
    return vocab, stoi


"""
    Convierte un DataFrame de textos y etiquetas en tensores usando el tokenizador de BERT.

    Esta función toma un DataFrame con columnas de texto y etiquetas, tokeniza cada texto,
    lo convierte en una secuencia de IDs de tokens, y ajusta la longitud de cada secuencia
    (rellenando con el token <pad> si es necesario). También convierte las etiquetas en tensores.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las columnas de texto y etiqueta.
    stoi : dict
        Diccionario de mapeo de token a índice del vocabulario de BERT.
    max_len : int
        Longitud máxima de las secuencias.
    text_col : str, opcional (por defecto="Body")
        Nombre de la columna que contiene el texto.
    label_col : str, opcional (por defecto="label")
        Nombre de la columna que contiene las etiquetas.

    Retorna
    -------
    X : torch.LongTensor
        Tensor con los índices de tokens, rellenados/truncados a `max_len`.
    y : torch.LongTensor
        Tensor con las etiquetas.
"""

def text_to_tensor(df, stoi, max_len, text_col="Body", label_col="label"):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    pad_idx = tokenizer.pad_token_id
    unk_idx = tokenizer.unk_token_id

    def to_ids(text):
        # Codifica el texto en una secuencia de IDs de tokens, añade tokens especiales,
        # y ajusta la longitud a max_len (rellena con <pad> si es necesario).
        ids = tokenizer.encode(text, add_special_tokens=True, max_length=max_len, truncation=True)
        if len(ids) < max_len:
            ids += [pad_idx] * (max_len - len(ids))
        return torch.tensor(ids[:max_len], dtype=torch.long)

    # Aplica la función de conversión a cada texto del DataFrame
    X = torch.stack([to_ids(t) for t in df[text_col]])
    # Convierte las etiquetas en tensor
    y = torch.tensor(df[label_col].values, dtype=torch.long)
    return X, y