
import torch
import argparse
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import get_device, preprocess_text, SummarizationDataset

def main(model_path, data_path):
    # Load Model and Tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    # Device
    device = get_device()
    model.to(device)

    # Load Dataset
    dataset = pd.read_csv(data_path)
    val_texts, val_summaries = dataset['article'], dataset['summary']
    val_encodings = preprocess_text(tokenizer, val_texts.tolist(), val_summaries.tolist(), max_length=512)

    # DataLoader
    val_dataset = SummarizationDataset(val_encodings)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Evaluation Function
    def evaluate(model, data_loader, device):
        model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    # Evaluate
    avg_loss = evaluate(model, val_loader, device)
    print(f'Average Loss: {avg_loss}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing validation data')
    args = parser.parse_args()
    main(args.model_path, args.data_path)
