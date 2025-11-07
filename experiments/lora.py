from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch


# ============================================================
# 0. PARÁMETROS CENTRALIZADOS
# ============================================================

# -----------------------------
# 🧩 Parámetros de LoRA
# -----------------------------
LORA_R = 32                    # rango bajo → define los “grados de libertad” adicionales (dimensión adaptativa)
LORA_ALPHA = 64                # factor de escala interna → amplifica la actualización de los adaptadores
LORA_DROPOUT = 0.05            # probabilidad de apagar adaptadores (regularización → evita overfitting)
LORA_TARGET_MODULES = ["query", "value"]  # subcapas de atención donde se aplica LoRA (afecta Q y V)
LORA_BIAS = "none"             # no se modifican los bias del modelo base

# -----------------------------
# ⚙️ Parámetros de entrenamiento
# -----------------------------
LEARNING_RATE = 5e-5           # tasa de aprendizaje → tamaño del paso en descenso de gradiente
BATCH_SIZE = 4                 # cantidad de ejemplos por batch (impacta en estabilidad y memoria)
GRAD_ACCUM_STEPS = 4           # pasos para acumular gradientes antes de actualizar
NUM_EPOCHS = 2                 # número de pasadas completas sobre el dataset
WEIGHT_DECAY = 0.01            # penalización de pesos grandes → regularización L2
LR_SCHEDULER = "linear"        # plan de ajuste del learning rate a lo largo del entrenamiento
WARMUP_RATIO = 0.1             # porcentaje inicial de pasos donde se incrementa gradualmente el LR
LABEL_SMOOTH = 0.1             # suavizado de etiquetas → evita sobreconfianza
LOGGING_STEPS = 20             # frecuencia de registro de métricas
FP16 = torch.cuda.is_available()  # habilita precisión mixta en GPU (optimiza rendimiento)
EVAL_STRATEGY = "epoch"        # evalúa al final de cada época
OUTPUT_DIR = "../data/processed/results"  # carpeta de salida


# ============================================================
# 1. Función para cargar dataset desde Excel
# ============================================================
def load_dataset(path, text_col="Body", label_col="Scores", seed=42, test_size=0.10):
    df = pd.read_excel(path)
    df = df.dropna(subset=[text_col])

    # Convertimos Scores a int y mapeamos a 0..K-1
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    unique_classes = sorted(df[label_col].unique())
    cls2id = {c: i for i, c in enumerate(unique_classes)}
    df["label"] = df[label_col].map(cls2id).astype(int)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), len(unique_classes)


# ============================================================
# 2. Rutas y carga de datos
# ============================================================
CURRENT_DIR = "../data/processed"
DATA_PATH = CURRENT_DIR + "/Abstracts.xlsx"

train_df, test_df, num_labels = load_dataset(path=str(DATA_PATH), text_col="Body", label_col="Scores")


# ============================================================
# 3. Tokenizador y modelo base
# ============================================================
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# ============================================================
# 4. Configuración LoRA
# ============================================================
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS
)
model = get_peft_model(model, lora_config)


# ============================================================
# 5. Tokenización del texto
# ============================================================
def tokenize(batch):
    return tokenizer(batch["Body"], truncation=True, padding="max_length", max_length=512)

train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)


# ============================================================
# 6. Función de métricas (accuracy + F1)
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


# ============================================================
# 7. Argumentos de entrenamiento
# ============================================================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy=EVAL_STRATEGY,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    label_smoothing_factor=LABEL_SMOOTH,
    logging_steps=LOGGING_STEPS,
    fp16=FP16,
)


# ============================================================
# 8. Entrenador
# ============================================================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
