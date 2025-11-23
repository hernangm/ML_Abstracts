import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path, text_col="Body", label_col="Scores", seed=42, test_size=0.1):
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

