import torch
from text_preprocessor.tokenizer_bert import build_vocab, text_to_tensor
from transformers import BertForSequenceClassification

"""
    Prepara los datos, el tokenizador y el modelo para un clasificador basado en transformers (BERT).

    Esta función realiza los siguientes pasos:
    1. Carga el vocabulario de BERT usando la función build_vocab, que obtiene la lista de tokens y el diccionario de índices.
    2. Tokeniza y convierte los textos de los conjuntos de entrenamiento y prueba en tensores de índices, usando la función text_to_tensor.
    3. Inicializa el modelo BertForSequenceClassification, que es una arquitectura de BERT adaptada para tareas de clasificación de texto.
       El modelo se carga desde el checkpoint "bert-base-uncased" y se ajusta el número de clases (num_labels) según la configuración.
    4. Devuelve el modelo, el vocabulario, el diccionario de índices, y los tensores de datos y etiquetas para entrenamiento y prueba.

    Parámetros
    ----------
    df_train : pandas.DataFrame
        DataFrame de entrenamiento que contiene las columnas de texto y etiqueta.
    df_test : pandas.DataFrame
        DataFrame de prueba que contiene las columnas de texto y etiqueta.
    cfg : objeto de configuración
        Debe tener los siguientes atributos:
        - TEXT_COL: nombre de la columna de texto.
        - MAX_LEN: longitud máxima de las secuencias.
        - NUM_CLASSES: número de clases de salida.
        - DEVICE: dispositivo de cómputo (cpu o cuda).

    Retorna
    -------
    model : BertForSequenceClassification
        Modelo BERT inicializado para clasificación de secuencias.
    vocab : list of str
        Lista de tokens del vocabulario de BERT.
    stoi : dict
        Diccionario que mapea cada token a su índice entero.
    X_train : torch.Tensor
        Tensor con las secuencias de IDs de tokens para los datos de entrenamiento.
    y_train : torch.Tensor
        Tensor con las etiquetas de entrenamiento.
    X_test : torch.Tensor
        Tensor con las secuencias de IDs de tokens para los datos de prueba.
    y_test : torch.Tensor
        Tensor con las etiquetas de prueba.
"""

def prepare_transformer_model(df_train, df_test, cfg):
    # Paso 1: Cargar el vocabulario de BERT
    print("Loading BERT vocabulary...")
    vocab, stoi = build_vocab()

    # Paso 2: Tokenizar y convertir los textos en tensores
    print("Tokenizing and converting text to tensors...")
    X_train, y_train = text_to_tensor(df_train, stoi, cfg.MAX_LEN, text_col=cfg.TEXT_COL, label_col="label")
    X_test, y_test = text_to_tensor(df_test, stoi, cfg.MAX_LEN, text_col=cfg.TEXT_COL, label_col="label")

    # Paso 3: Inicializar el modelo BERT para clasificación
    print("Initializing BERT model for sequence classification...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=cfg.NUM_CLASSES
    ).to(cfg.DEVICE)

    # Paso 4: Retornar todos los objetos necesarios para el entrenamiento
    return model, vocab, stoi, X_train, y_train, X_test, y_test