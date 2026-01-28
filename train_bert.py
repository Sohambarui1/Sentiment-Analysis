# ============================================================
# CPU-OPTIMIZED DISTILBERT FOR i3 (Low Heat • Max Accuracy)
# Light on Resources • Smart Training Strategy
# ============================================================

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import re, os, warnings

warnings.filterwarnings("ignore")

# ============================================================
# 🔧 CPU OPTIMIZATION (i3 SAFE - NO OVERHEATING)
# ============================================================
torch.set_num_threads(2)           # 2 threads for i3 (4 cores)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force CPU only, disable GPU
device = torch.device("cpu")
torch.set_float32_matmul_precision('medium')  # Reduce precision for speed

# ============================================================
# ⚙️ CONFIG - OPTIMIZED FOR i3 CPU
# ============================================================
DATA_PATH = "data/Combined.csv"
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 96           # Shorter sequences = faster processing
BATCH_SIZE = 8         # Small batch for low memory footprint
EPOCHS = 4             # Smart epochs (early stopping if needed)
LR = 2e-5              # Conservative learning rate
WEIGHT_DECAY = 0.01    # L2 regularization
GRADIENT_ACCUMULATION = 2  # Simulate larger batch without memory overhead

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

# Filter: keep all that exist in data
existing_labels = df["label"].unique()
VALID_LABELS = [label for label in VALID_LABELS if label in existing_labels]

print(f"\n📋 Valid labels found in dataset: {VALID_LABELS}")

df = df[df["label"].isin(VALID_LABELS)]
df = df[df["text"].str.len() >= 10]  # Lower threshold for more data

print("\n📊 Original label distribution:")
print(df["label"].value_counts())

# -----------------------------
# SMART BALANCING WITH DATA AUGMENTATION
# -----------------------------
# Use stratified sampling with higher max per class to preserve more data
min_class_size = df.groupby("label").size().min()
MAX_PER_CLASS = 1500  # Optimized for i3 CPU

print(f"\n⚖️ Min class size: {min_class_size}, Max per class: {MAX_PER_CLASS}")

balanced = []

for lbl in df["label"].unique():
    sub = df[df["label"] == lbl]
    if len(sub) > MAX_PER_CLASS:
        sub = sub.sample(MAX_PER_CLASS, random_state=42)
    balanced.append(sub)

df = pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)

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

# ============================================================
# MODEL SETUP - OPTIMIZED FOR i3 CPU
# ============================================================
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES
)

# 🔑 SMART FREEZING: Freeze first 2 layers (saves computation on i3)
for layer in model.distilbert.transformer.layer[:2]:
    for p in layer.parameters():
        p.requires_grad = False

model.to(device)
print("\n✅ Model loaded on CPU (i3 optimized)")

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# Learning rate scheduler for smooth convergence
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ============================================================
# MAIN TRAINING LOOP - i3 CPU OPTIMIZED
# ============================================================
best_f1 = 0.0
best_accuracy = 0.0
patience = 2
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    print(f"\n🔹 Epoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0
    train_steps = 0
    accumulation_step = 0

    for batch_idx, batch in enumerate(train_loader):
        # Gradient accumulation for stable learning on i3
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / GRADIENT_ACCUMULATION
        loss.backward()
        
        accumulation_step += 1
        train_steps += 1
        total_loss += outputs.loss.item()

        # Optimizer step after accumulation
        if accumulation_step == GRADIENT_ACCUMULATION:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            accumulation_step = 0

    scheduler.step()

    # ============================================================
    # VALIDATION PHASE
    # ============================================================
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            true.extend(batch["labels"].cpu().numpy())

    f1 = f1_score(true, preds, average="macro")
    accuracy = accuracy_score(true, preds)
    avg_loss = total_loss / train_steps
    
    print(f"✓ Loss: {avg_loss:.4f} | Accuracy: {accuracy*100:.2f}% | F1: {f1*100:.2f}%")

    # Early stopping with patience
    if f1 > best_f1:
        best_f1 = f1
        best_accuracy = accuracy
        epochs_without_improvement = 0
        
        # Save model with safe file handling
        try:
            os.makedirs("bert_model", exist_ok=True)
            model.save_pretrained("bert_model", safe_serialization=False)
            tokenizer.save_pretrained("bert_model")
            np.save("label_classes.npy", label_encoder.classes_)
            print("✅ Best model saved!")
        except Exception as e:
            print(f"⚠️ Error saving model: {str(e)}")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"\n⏹️ Early stopping after {epoch+1} epochs (no F1 improvement for {patience} epochs)")
            break

# -----------------------------
# FINAL REPORT
# -----------------------------
# ============================================================
# FINAL COMPREHENSIVE REPORT
# ============================================================
print("\n" + "="*70)
print("📋 FINAL CLASSIFICATION REPORT (All Conditions)")
print("="*70)
print(classification_report(true, preds, target_names=label_encoder.classes_, digits=4))

print("\n📊 CONFUSION MATRIX ANALYSIS:")
cm = confusion_matrix(true, preds)
print(cm)

print("\n" + "="*70)
print("🏁 TRAINING SUMMARY")
print("="*70)
print(f"✅ Best Macro F1 Score: {best_f1*100:.2f}%")
print(f"✅ Best Accuracy: {best_accuracy*100:.2f}%")
print(f"📊 Total Classes: {NUM_CLASSES}")
print(f"📈 Classes: {list(label_encoder.classes_)}")
print(f"📁 Model saved to: bert_model/")
print("="*70)
