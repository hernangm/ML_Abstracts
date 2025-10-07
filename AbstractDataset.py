from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModel

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AbstractDataset(Dataset):
    def __init__(self, texts, targets, max_length=128):
        self.texts = texts.tolist()
        self.targets = torch.tensor(targets.values, dtype=torch.float32).view(-1, 1)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask, target