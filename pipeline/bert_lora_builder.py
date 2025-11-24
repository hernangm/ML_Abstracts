import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

def prepare_bert_lora_model(df_train, df_test, cfg):
    """
    Prepares tokenizer, encoded inputs, BERT model augmented LoRA.

    Parameters
    df_train : DataFrame
        Training samples.
    df_test : DataFrame
        Test samples.
    cfg : Config
        Configuration object.

    Key arguments
    model_name : pretrained checkpoint.
    max_length : sequence cap.
    batch_size : training size.
    num_labels : output classes.
    r : LoRA rank.
    lora_alpha : scaling factor.
    lora_dropout : LoRA dropout.
    device : compute target.

    """
    model_name = "bert-base-multilingual-cased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # encode texts
    def encode_texts(series):
        enc = tokenizer(
            list(series),
            padding="max_length",
            truncation=True,
            max_length=cfg.MAX_LEN,
            return_tensors="pt"
        )
        return enc["input_ids"], enc["attention_mask"]

    X_train_ids, X_train_mask = encode_texts(df_train[cfg.TEXT_COL])
    X_test_ids, X_test_mask = encode_texts(df_test[cfg.TEXT_COL])

    y_train = torch.tensor(df_train["label"].values, dtype=torch.long)
    y_test = torch.tensor(df_test["label"].values, dtype=torch.long)

    # ---- load base model ----
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=cfg.NUM_CLASSES
    )

    # ---- LoRA ----
    lora_cfg = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_cfg)
    model.to(cfg.DEVICE)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "X_train_ids": X_train_ids,
        "X_train_mask": X_train_mask,
        "y_train": y_train,
        "X_test_ids": X_test_ids,
        "X_test_mask": X_test_mask,
        "y_test": y_test,
    }
