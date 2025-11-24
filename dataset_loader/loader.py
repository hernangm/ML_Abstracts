import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path, text_col="Body", label_col="Scores", seed=42, test_size=0.1):
    """
    Loads an Excel dataset
    cleans text rows
    encodes labels and performs
    stratified split

    Parameters
    path : str
        File path.
    text_col : str
        Text column name.
    label_col : str
        Label column name.
    seed : int
        RNG seed.
    test_size : float
        Test split ratio.

    Key steps
    dropna : remove empty texts.
    to_numeric : cast labels.
    cls2id : class mapping.
    stratify : balanced split.

    Returns
    train_df : DataFrame
        Training subset.
    test_df : DataFrame
        Test subset.
    num_classes : int
        Count of unique labels.
    """
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

