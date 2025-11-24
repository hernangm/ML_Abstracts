# Clasificador de Resúmenes Médicos con Redes Neuronales Recurrentes

Este proyecto implementa y compara varias arquitecturas de redes
neuronales para la **clasificación de resúmenes médicos**, incluyendo
RNNs tradicionales, variantes avanzadas y modelos basados en
transformers. También soporta Fine-tuning con LoRA para entrenar
modelos grandes de forma eficiente.

## Descripción General

El sistema está diseñado para ser **modular** y **extensible**,
permitiendo experimentar con diferentes arquitecturas. Actualmente
soporta:

- **RNN (Recurrent Neural Network):** Arquitectura secuencial básica.
- **LSTM (Long Short-Term Memory):** Captura dependencias largas en el
    texto.
- **GRU (Gated Recurrent Unit):** Variante simplificada de LSTM.
- **Transformer (BERT):** Utiliza `bert-base-uncased` para
    clasificación de texto.
- **Modelos con Fine-tuning LoRA:** Los modelos basados en RNN (RNN,
    LSTM, GRU) pueden activarse con LoRA para reducir el costo
    computacional, entrenando solo algunos parámetros adicionales.

### Flujo principal del sistema

1. Cargar el dataset desde un archivo Excel.
2. Preprocesar texto y etiquetas.
3. Construir vocabulario (RNN) o cargar tokenizador (BERT).
4. Seleccionar arquitectura (con o sin LoRA).
5. Entrenar y evaluar.

## Fine-tuning con LoRA

**LoRA (Low-Rank Adaptation)** permite entrenar modelos grandes
agregando solo un pequeño número de parámetros adicionales, lo que
ofrece:

- menor uso de memoria,
- entrenamiento más rápido,
- posibilidad de usar hardware limitado.

**Implementación:** `utils/lora_utils.py`\

## Estructura del Proyecto

``` text
.
├── main.py                       # Menú interactivo
├── data/
│   ├── raw/                      # Datos originales
│   └── processed/                # Datos limpios (e.g., Abstracts.xlsx)
├── dataset_loader/
│   └── loader.py                 # Carga y preprocesamiento del dataset
├── models/
│   ├── rnn_classifier.py         # RNN / RNN + LoRA
│   ├── lstm_classifier.py        # LSTM / LSTM + LoRA
│   ├── gru_classifier.py         # GRU / GRU + LoRA
│   └── transformer_classifier.py # BERT
├── pipeline/
│   ├── text_model_builder.py        # Construcción de modelos RNN
│   └── transformer_model_builder.py # Construcción del modelo BERT
├── train/
│   ├── trainer.py                 # Entrenamiento
│   └── evaluator.py               # Evaluación
└── utils/
    ├── config.py                  # Configuración global
    └── lora_utils.py              # Implementación de LoRA
```

## Cómo Ejecutar el Proyecto

### Requisitos

``` bash
pip install torch pandas scikit-learn transformers
```

### Ejecución

``` bash
python main.py
```

Aparecerá el menú:

``` text
Seleccione el modelo a entrenar:
1) rnn
2) lstm
3) gru
4) rnn_scheduler
5) rnn_phrases
6) transformer
0) salir
```

Seleccionar el modelo para iniciar el preprocesamiento, entrenamiento y
evaluación.
