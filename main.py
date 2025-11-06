import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_loader.loader import load_dataset
from pipeline.text_model_builder import prepare_text_model
from train.trainer import train_model
from train.evaluator import evaluate_model
from utils.config import Config





if __name__ == "__main__":
    cfg = Config()

    print("Cargando dataset...")
    df_train, df_test, num_classes = load_dataset(cfg.DATA_PATH, text_col=cfg.TEXT_COL, label_col=cfg.LABEL_COL)
    cfg.NUM_CLASSES = num_classes

    model, vocab, stoi, X_train, y_train, X_test, y_test = prepare_text_model(df_train, df_test, cfg)

    print(f"Entrenando modelo {cfg.MODEL_TYPE.upper()}...")
    train_model(model, X_train, y_train, cfg)

    print(f"📈 Evaluando modelo {cfg.MODEL_TYPE.upper()}...")
    evaluate_model(model, X_test, y_test, cfg)
