import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ----------------------------
# Helper functions
# ----------------------------
def pad_sequences(sequences, maxlen, padding='post', truncating='post', value=0):
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - len(seq)
            if padding == 'post':
                seq = seq + [value] * pad_len
            else:
                seq = [value] * pad_len + seq
        padded.append(seq)
    return np.array(padded)

# ----------------------------
# Load dataset
# ----------------------------
data = pd.read_csv("data/Combined.csv")
data = data.dropna(subset=["statement", "status"])
data = data.head(1000)  # Use only first 1000 rows for testing

texts = data["statement"].astype(str).tolist()
labels = data["status"].astype(str).tolist()

# ----------------------------
# Encode labels
# ----------------------------
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

num_classes = len(set(encoded_labels))

# ----------------------------
# Simple Tokenizer
# ----------------------------
class SimpleTokenizer:
    def __init__(self, num_words=None, oov_token=None):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}
        self.index_word = {}
        self.word_counts = {}
    
    def fit_on_texts(self, texts):
        for text in texts:
            for word in text.split():
                if word not in self.word_counts:
                    self.word_counts[word] = 0
                self.word_counts[word] += 1
        
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        if self.num_words:
            sorted_words = sorted_words[:self.num_words-1]  # reserve 1 for OOV
        
        self.word_index = {word: i+1 for i, (word, _) in enumerate(sorted_words)}
        if self.oov_token:
            self.word_index[self.oov_token] = len(self.word_index) + 1
        self.index_word = {v: k for k, v in self.word_index.items()}
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = []
            for word in text.split():
                if word in self.word_index:
                    seq.append(self.word_index[word])
                elif self.oov_token and self.oov_token in self.word_index:
                    seq.append(self.word_index[self.oov_token])
                else:
                    pass  # ignore unknown
            sequences.append(seq)
        return sequences

# ----------------------------
# Tokenization
# ----------------------------
MAX_WORDS = 20000
MAX_LEN = 100

tokenizer = SimpleTokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN)

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, encoded_labels, test_size=0.2, random_state=42
)

# ----------------------------
# LSTM Model
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout_rate):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        output = self.fc(hidden)
        return output

model = LSTMModel(MAX_WORDS, 128, 128, num_classes, 0.4)

# ----------------------------
# Loss and optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# ----------------------------
# Convert to tensors
# ----------------------------
X_train = torch.tensor(X_train, dtype=torch.long)
y_train_indices = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)
y_test_indices = torch.tensor(y_test, dtype=torch.long)

# ----------------------------
# DataLoader
# ----------------------------
train_dataset = TensorDataset(X_train, y_train_indices)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ----------------------------
# Train
# ----------------------------
for epoch in range(5):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/5 completed")

# ----------------------------
# Save model & tokenizer
# ----------------------------
try:
    torch.save(model.state_dict(), "lstm_model.pth")
    print("Model saved successfully")
except Exception as e:
    print(f"Failed to save model: {e}")

try:
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer saved successfully")
except Exception as e:
    print(f"Failed to save tokenizer: {e}")

try:
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved successfully")
except Exception as e:
    print(f"Failed to save label encoder: {e}")

print("✅ LSTM model trained and saved")          