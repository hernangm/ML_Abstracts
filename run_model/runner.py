from dataset_loader.loader import load_dataset
from pipeline.text_model_builder import prepare_text_model
from train.trainer import train_model
from train.evaluator import evaluate_model

# BERT + LoRA
from pipeline.bert_lora_builder import prepare_bert_lora_model
from pipeline.bert_lora_train import train_bert_lora
from pipeline.bert_lora_eval import evaluate_bert_lora


def run_model(cfg):
    """
    Orchestrates dataset loading
    model construction
    training
    evaluation
    for both classic sequence models RNN-LSTM-GRU and BERT+LoRA.

    Parameters
    cfg : Config
        Global configuration.

    Key arguments
    data_path : dataset location.
    text_col : input text field.
    label_col : raw label field.
    num_classes : output size.
    model_type : architecture selector.
    X_train / y_train : training tensors.
    X_test / y_test : evaluation tensors.
    scheduler : LR scheduler (optional).
    bert_lora : transformer fine-tuning path.

    Main steps
    load_dataset : import and split data.
    prepare_text_model : build vocab + init model.
    prepare_bert_lora_model : tokenizer + BERT-LoRA setup.
    train_model : train classical nets.
    train_bert_lora : fine-tune transformer.
    evaluate_model : final metrics.
    evaluate_bert_lora : transformer accuracy.

        Executes the full training/evaluation workflow based  cfg.MODEL_TYPE.
    """

    print("Cargando dataset...")
    df_train, df_test, num_classes = load_dataset(
        cfg.DATA_PATH,
        text_col=cfg.TEXT_COL,
        label_col=cfg.LABEL_COL,
    )
    cfg.NUM_CLASSES = num_classes

    model_type = cfg.MODEL_TYPE
    print(f"Preparando modelo ({model_type})...")

    #  BERT + LoRA
    if model_type == "bert_lora":
        pack = prepare_bert_lora_model(df_train, df_test, cfg)

        train_bert_lora(
            model=pack["model"],
            X_ids=pack["X_train_ids"],
            X_mask=pack["X_train_mask"],
            y=pack["y_train"],
            cfg=cfg
        )

        print("\nEvaluando modelo BERT + LoRA...")
        evaluate_bert_lora(
            model=pack["model"],
            X_ids=pack["X_test_ids"],
            X_mask=pack["X_test_mask"],
            y=pack["y_test"],
            cfg=cfg
        )
        return

    prepared = prepare_text_model(df_train, df_test, cfg)

    if model_type == "rnn_scheduler":
        (
            model, vocab, stoi,
            X_train, y_train,
            X_test, y_test,
            scheduler
        ) = prepared

        train_model(
            model, X_train, y_train, cfg,
            scheduler=scheduler,
            X_test=X_test, y_test=y_test
        )

    else:
        (
            model, vocab, stoi,
            X_train, y_train,
            X_test, y_test
        ) = prepared

        train_model(
            model, X_train, y_train, cfg,
            X_test=X_test, y_test=y_test
        )

    print("\nEvaluando modelo final...")
    evaluate_model(model, X_test, y_test, cfg)
