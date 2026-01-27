# ==============================
# FINAL LSTM Inference Flask App
# ==============================

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import dill
import re
from textblob import TextBlob

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "lstm_model.pth"
TOKENIZER_PATH = "tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
MAX_LEN = 100

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Tokenizer (MUST MATCH TRAINING)
# ----------------------------
class SimpleTokenizer:
    def __init__(self, oov_token="<OOV>"):
        self.oov_token = oov_token
        self.word_index = {}

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [
                self.word_index.get(word, self.word_index[self.oov_token])
                for word in text.split()
            ]
            sequences.append(seq)
        return sequences

# ----------------------------
# LSTM Model
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, output_dim=7):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# ----------------------------
# Load Artifacts
# ----------------------------
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = dill.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = dill.load(f)

# MUST match training-time vocab size
VOCAB_SIZE = 20000

NUM_CLASSES = len(label_encoder.classes_)

model = LSTMModel(VOCAB_SIZE, output_dim=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ LSTM model and tokenizer loaded successfully")

# ----------------------------
# Helpers
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def pad_sequence(seq):
    seq = seq[:MAX_LEN]
    return seq + [0] * (MAX_LEN - len(seq))

def map_risk(category):
    if category in ["Suicidal", "Bipolar", "PTSD"]:
        return "High"
    elif category in ["Depression", "Anxiety", "Stress"]:
        return "Medium"
    return "Low"

def get_recommendation(category, risk):
    if category == "Suicidal":
        return (
            "🚨 High-risk mental health state detected. "
            "Immediate professional help is strongly recommended. "
            "If you are in danger, contact emergency services or a suicide prevention helpline immediately."
        )
    if category == "Bipolar":
        return "Consult a psychiatrist for mood stabilization and maintain a structured routine."
    if category == "PTSD":
        return "Trauma-focused therapy and professional counseling are recommended."
    if category == "Depression":
        return "Therapy, regular routine, physical activity, and social support can help."
    if category == "Anxiety":
        return "Mindfulness, breathing exercises, and professional guidance may help."
    if category == "Stress":
        return "Time management, adequate rest, and relaxation techniques are recommended."
    return "Maintain healthy habits and continue positive coping strategies."

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Invalid JSON input"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty input"}), 400

    # -------- LSTM Prediction --------
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])[0]
    padded = pad_sequence(seq)

    x = torch.tensor([padded], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = torch.argmax(probs).item()
        confidence = round(float(probs[idx]) * 100, 1)

    category = label_encoder.inverse_transform([idx])[0].capitalize()
    risk = map_risk(category)

    # -------- Sentiment --------
    sentiment_score = round(TextBlob(text).sentiment.polarity, 3)
    if sentiment_score > 0.2:
        sentiment_label = "Positive"
    elif sentiment_score < -0.2:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    # -------- Recommendation --------
    recommendation = get_recommendation(category, risk)

    return jsonify({
        "category": category,
        "risk": risk,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "confidence": confidence,
        "word_count": len(text.split()),
        "recommendation": recommendation
    })

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
