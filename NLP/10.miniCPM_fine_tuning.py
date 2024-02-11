import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # for progress bars
import numpy as np  # Necessary for handling logits and labels for accuracy calculation

#----------------------------Data Preparation-------------------------#
# Load the dataset
df = pd.read_csv('data/DMSC.csv')

# Map star ratings to sentiment labels (0=negative, 1=positive, 2=neutral)
def label_sentiment(row):
    if row['Star'] >= 4:  # if higher than 4, then it's positive
        return 1  # Positive
    elif row['Star'] <= 2:
        return 0  # Negative
    else:
        return 2  # Neutral or consider dropping these rows

# Apply the mapping
df['Sentiment'] = df.apply(label_sentiment, axis=1)

# Dropping neutral for binary classification
df = df[df['Sentiment'] != 2]

# Preprocess comments
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Remove all non-Chinese and non-basic punctuation characters for Chinese text
    text = re.sub(r'[^\u4e00-\u9fff，。！？、]+', ' ', text)
    # Trim leading and trailing whitespaces
    text = text.strip()
    return text

df['Processed_Comment'] = df['Comment'].apply(preprocess_text)

#---------------------------------LLM---------------------------------#

tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-2B-dpo-bf16-llama-format")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set correctly

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
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Convert dataframe to dataset
dataset = SentimentDataset(df['Processed_Comment'].to_list(), df['Sentiment'].to_list(), tokenizer)

# Create DataLoader
loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

#---------------------------------Model---------------------------------#
class MiniCPMForSentimentAnalysis(nn.Module):
    def __init__(self, model_name, num_labels):
        super(MiniCPMForSentimentAnalysis, self).__init__()
        self.mini_cpm = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.mini_cpm.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.mini_cpm(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
# Initialize the custom model
model = MiniCPMForSentimentAnalysis("openbmb/MiniCPM-2B-dpo-bf16-llama-format", num_labels=2)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#---------------------------------Training---------------------------------#

def train(model, data_loader, optimizer, device, num_epochs=3):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        predictions, true_labels = [], []

        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            logits = outputs.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            batch_predictions = np.argmax(logits, axis=1)
            predictions.extend(batch_predictions)
            true_labels.extend(label_ids)

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Clearing the cache after each epoch
        torch.cuda.empty_cache()

# Train the model
optimizer = AdamW(model.parameters(), lr=5e-5)
train(model, loader, optimizer, device, num_epochs=3)