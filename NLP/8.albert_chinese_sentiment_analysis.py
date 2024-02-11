import torch
import torch.nn as nn
from transformers import BertTokenizer, AlbertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np


#----------------------------Data Preparation-------------------------#
# Load the dataset
df = pd.read_csv('data/DMSC.csv')
# Randomly sample 10,000 rows from the dataset
df = df.sample(n=10000, random_state=42)

# Map star ratings to sentiment labels (0=negative, 1=positive, 2=neutral)
def label_sentiment(row):
    if row['Star'] >= 4:
        return 1
    elif row['Star'] <= 2:
        return 0
    else:
        return 2

df['Sentiment'] = df.apply(label_sentiment, axis=1)
df = df[df['Sentiment'] != 2]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fff，。！？、]+', ' ', text)
    text = text.strip()
    return text

df['Processed_Comment'] = df['Comment'].apply(preprocess_text)

# Splitting data
df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
df_train, df_val = train_test_split(df_train, test_size=(2/9), random_state=42)  # Approx. 20% for validation

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_base')
model = AlbertForSequenceClassification.from_pretrained('voidful/albert_chinese_base', num_labels=3)

class SentimentDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = comments
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text, None, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', return_token_type_ids=False,
            truncation=True, return_attention_mask=True, return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Datasets and DataLoaders
train_dataset = SentimentDataset(df_train['Processed_Comment'].tolist(), df_train['Sentiment'].tolist(), tokenizer)
val_dataset = SentimentDataset(df_val['Processed_Comment'].tolist(), df_val['Sentiment'].tolist(), tokenizer)
test_dataset = SentimentDataset(df_test['Processed_Comment'].tolist(), df_test['Sentiment'].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
model = AlbertForSequenceClassification.from_pretrained('voidful/albert_chinese_base', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training
def train(model, data_loader, optimizer, device, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        total_loss, predictions, true_labels = 0, [], []
        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            batch_predictions = np.argmax(logits, axis=1)
            predictions.extend(batch_predictions)
            true_labels.extend(label_ids)
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss/len(data_loader):.4f}, Accuracy: {accuracy:.4f}")
        torch.cuda.empty_cache()  # Clear memory cache

optimizer = AdamW(model.parameters(), lr=5e-5)
train(model, train_loader, optimizer, device)

# Evaluation
def evaluate(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())
    return accuracy_score(true_labels, predictions)

val_accuracy = evaluate(model, val_loader, device)
test_accuracy = evaluate(model, test_loader, device)
print(f'Validation Accuracy: {val_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Save the model and tokenizer
model_save_path = './model/albert_chinese_sentiment'
tokenizer_save_path = './model/albert_chinese_sentiment'

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)