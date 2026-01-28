# ============================================================
# FINAL SINGLE-STAGE DISTILBERT TRAINING
# Accurate • No Label Leakage • i3 + 8GB RAM Safe
# ============================================================

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import re, os, warnings

warnings.filterwarnings("ignore")

# -----------------------------
# CPU SAFETY (IMPORTANT)
# -----------------------------
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/Combined.csv"
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 96
BATCH_SIZE = 8
EPOCHS = 4           # minimum for real learning
LR = 2e-5

# -----------------------------
# CLEAN TEXT (NO LEAKAGE)
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s.,!?']", "", text)
    return re.sub(r"\s+", " ", text).strip()

# -----------------------------
# LOAD DATA (RAW LABELS ONLY)
# -----------------------------
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["statement", "status"])

df["text"] = df["statement"].apply(clean_text)
df["label"] = df["status"].astype(str).str.strip()

# Keep only valid dataset labels
VALID_LABELS = [
    "Anxiety",
    "Depression",
    "Stress",
    "Bipolar",
    "Personality_disorder",
    "PTSD",
    "Suicidal",
    "Normal",
    "Well-being"
]

df = df[df["label"].isin(VALID_LABELS)]
df = df[df["text"].str.len() >= 15]

print("\n📊 Original label distribution:")
print(df["label"].value_counts())

# -----------------------------
# SAFE BALANCING (NO OVERSHORTCUT)
# -----------------------------
MAX_PER_CLASS = 1200
balanced = []

for lbl in df["label"].unique():
    sub = df[df["label"] == lbl]
    if len(sub) > MAX_PER_CLASS:
        sub = sub.sample(MAX_PER_CLASS, random_state=42)
    balanced.append(sub)

df = pd.concat(balanced).sample(frac=1, random_state=42)

print("\n📊 Balanced label distribution:")
print(df["label"].value_counts())

# -----------------------------
# LABEL ENCODING
# -----------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])
NUM_CLASSES = len(label_encoder.classes_)

print("\n✅ Classes:", list(label_encoder.classes_))

# -----------------------------
# TRAIN / VALIDATION SPLIT
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    df["text"].tolist(),
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# TOKENIZER & DATASET
# -----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_ds = TextDataset(X_train, y_train)
val_ds = TextDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# -----------------------------
# MODEL SETUP
# -----------------------------
device = torch.device("cpu")

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES
)

# 🔑 Freeze only first 2 layers (keeps accuracy)
for layer in model.distilbert.transformer.layer[:2]:
    for p in layer.parameters():
        p.requires_grad = False

model.to(device)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# -----------------------------
# TRAINING LOOP
# -----------------------------
best_f1 = 0.0

for epoch in range(EPOCHS):
    print(f"\n🔹 Epoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()

    # ----- VALIDATION -----
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            true.extend(batch["labels"].cpu().numpy())

    f1 = f1_score(true, preds, average="macro")
    print(f"Train Loss: {total_loss/len(train_loader):.4f} | Val Macro F1: {f1*100:.2f}%")

    if f1 > best_f1:
        best_f1 = f1
        os.makedirs("bert_model", exist_ok=True)
        model.save_pretrained("bert_model")
        tokenizer.save_pretrained("bert_model")
        np.save("label_classes.npy", label_encoder.classes_)
        print("✅ Best model saved")

# -----------------------------
# FINAL REPORT
# -----------------------------
print("\n📋 Final Classification Report:")
print(classification_report(true, preds, target_names=label_encoder.classes_))

print("\n🏁 Training completed")
print(f"🏆 Best Macro F1: {best_f1*100:.2f}%")
print("📁 Model saved to: bert_model/")
