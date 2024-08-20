
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer

class SummarizationDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_text(tokenizer, texts, summaries, max_length):
    input_encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    summary_encodings = tokenizer(
        summaries,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': summary_encodings['input_ids']
    }
