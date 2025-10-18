import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset_scores(path, text_col="Body", score_col="Scores", seed=42, test_size=0.2):
    # Lee la hoja 'ABSTRACTS' si existe; si no, lee la primera
    try:
        df = pd.read_excel(path, sheet_name="ABSTRACTS")
    except Exception:
        df = pd.read_excel(path)

    # Texto válido
    df = df.dropna(subset=[text_col])
    # Scores a enteros; si hay algo raro -> 0
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0).astype(int)

    # Mapeo a clases 0..K-1 (¡sin cambiar el nombre de columna!)
    classes = sorted(df[score_col].unique().tolist())
    cls2id = {c: i for i, c in enumerate(classes)}
    id2cls = {i: c for c, i in cls2id.items()}
    df["label"] = df[score_col].map(cls2id).astype(int)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), cls2id, id2cls
