import os
import json
import numpy as np
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score
import copy
from gensim.models import KeyedVectors

class GenreClassifierWithW2VEmbeddings(nn.Module):
    def __init__(self):
        super(GenreClassifierWithW2VEmbeddings, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(300 + self.bert.config.hidden_size, 4)

    def forward(self, input_ids, attention_mask, w2v_embeddings):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        avg_pooled = bert_outputs.last_hidden_state.mean(1)  # Average pooling

        # Ensure the w2v_embeddings tensor is of the same type as avg_pooled
        w2v_embeddings = w2v_embeddings.to(dtype=avg_pooled.dtype)

        concatenated = torch.cat((avg_pooled, w2v_embeddings), dim=1)  # Concatenate
        logits = self.classifier(concatenated)
        return logits


class GenreClassifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = GenreClassifierWithW2VEmbeddings()
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Load Word2Vec Model
        self.word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    def get_avg_word2vec(self, tokens_list, vector, k=300):
        """Compute the average word2vec for a list of tokens."""
        if len(tokens_list) < 1:
            return np.zeros(k)
        if vector == {}:
            return np.zeros(k)
        vecs = []
        for token in tokens_list:
            if token in vector:
                vec = vector[token]
                vecs.append(vec)
        if len(vecs) == 0:
            return np.zeros(k)
        vecs = np.array(vecs)
        return vecs.mean(axis=0)

    def get_word2vec_embeddings(self, text):
        tokens = self.tokenizer.tokenize(text)
        return self.get_avg_word2vec(tokens, self.word2vec)

    def preprocess_data(self, texts, labels=None):
        MAX_LENGTH = 512
        input_ids = []
        attention_masks = []

        for x in texts:
            encoded = self.tokenizer.encode_plus(x, max_length=MAX_LENGTH, padding='max_length', return_attention_mask=True, truncation=True)
            input_ids.append(encoded["input_ids"])
            attention_masks.append(encoded["attention_mask"])

        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        # For Word2Vec embeddings
        w2v_embeddings = [self.get_word2vec_embeddings(text) for text in texts]
        w2v_embeddings = torch.tensor(w2v_embeddings)

        if labels:
            labels = torch.tensor(labels)
            dataset = TensorDataset(input_ids, attention_masks, w2v_embeddings, labels)
        else:
            dataset = TensorDataset(input_ids, attention_masks, w2v_embeddings)

        return dataset

    def train(self, train_dataset, val_dataset, epochs=10, batch_size=4, lr=2e-5, max_grad_norm=1.0, patience=6):
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        validation_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*epochs)
        criterion = nn.CrossEntropyLoss()  # Define the criterion
        
        # For early stopping
        no_improve = 0
        best_f1 = 0
        best_model = None

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()

            for batch_idx, batch in enumerate(train_dataloader):
                input_ids, attention_mask, w2v_embeddings, labels = batch
                input_ids, attention_mask, w2v_embeddings, labels = input_ids.to(self.device), attention_mask.to(self.device), w2v_embeddings.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()

                logits = self.model(input_ids, attention_mask, w2v_embeddings)
                loss = criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                # F1 calculation can be optimized more if needed by accumulating and averaging later
                if (batch_idx + 1) % 50 == 0:
                    preds = torch.argmax(logits, dim=1)
                    train_f1 = f1_score(labels.cpu(), preds.cpu(), average='macro')
                    print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {loss.item()}, Train F1: {train_f1:.4f}")

            # Validation
            self.model.eval()
            eval_loss = 0
            all_preds = []
            all_labels = []

            for batch in validation_dataloader:
                input_ids, attention_mask, w2v_embeddings, labels = batch
                input_ids, attention_mask, w2v_embeddings, labels = input_ids.to(self.device), attention_mask.to(self.device), w2v_embeddings.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    logits = self.model(input_ids, attention_mask, w2v_embeddings)
                loss = criterion(logits, labels)
                eval_loss += loss.item()

                # Accumulate all predictions and labels for F1 calculation
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            avg_val_loss = eval_loss / len(validation_dataloader)
            eval_f1 = f1_score(all_labels, all_preds, average='macro')
            print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}, Validation F1: {eval_f1:.4f}")

            # Early stopping and model saving based on validation F1 score
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                best_model = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve == patience:
                    print(f"Early stopping after {patience} epochs with no improvement.")
                    self.model.load_state_dict(best_model)
                    break

    def predict(self, test_dataset, batch_size=16):
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
        self.model.eval()
        predictions = []

        for batch in test_dataloader:
            input_ids, attention_mask, w2v_embeddings = batch
            input_ids, attention_mask, w2v_embeddings = input_ids.to(self.device), attention_mask.to(self.device), w2v_embeddings.to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask, w2v_embeddings)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

        return predictions

if __name__ == "__main__":
    # Data Loading
    with open(os.path.join("data", "genre_train.json"), "r") as f:
        train_data = json.load(f)
    
    X_train = train_data['X']
    Y_train = train_data['Y']

    classifier = GenreClassifier()

    dataset = classifier.preprocess_data(X_train, Y_train)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    classifier.train(train_dataset, val_dataset)

    # Predictions on Test Set
    with open(os.path.join("data", "genre_test.json"), "r") as f:
        test_data = json.load(f)
    X_test = test_data['X']

    test_dataset = classifier.preprocess_data(X_test)
    predictions = classifier.predict(test_dataset)

    # Writing to CSV
    with open("out.csv", "w") as fout:
        fout.write("Id,Predicted\n")
        for i, pred in enumerate(predictions):
            fout.write(f"{i},{pred}\n")
