# =========================================
# FINAL DISTILBERT INFERENCE FLASK APP
# MATCHES train_bert.py (Single-Stage)
# =========================================

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import numpy as np
import re
import time
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from textblob import TextBlob
import os

# -----------------------------
# CONFIG (MUST MATCH TRAINING)
# -----------------------------
MODEL_DIR = "bert_model"
LABELS_PATH = "label_classes.npy"
MAX_LEN = 96   # ✅ MUST match training

# -----------------------------
# FLASK
# -----------------------------
app = Flask(__name__)
CORS(app)
device = torch.device("cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
print("🤖 Loading model...")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

label_classes = np.load(LABELS_PATH, allow_pickle=True)
label_classes = list(label_classes)

print("✅ Model loaded")
print("Classes:", label_classes)

# -----------------------------
# HELPERS
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sentiment_info(text):
    blob = TextBlob(text)
    return round(blob.sentiment.polarity, 3)

def risk_from_label(label):
    if label == "Suicidal":
        return "Critical"
    if label in ["Bipolar", "Personality_disorder"]:
        return "High"
    if label in ["Depression", "Anxiety", "Stress"]:
        return "Medium"
    if label == "Normal":
        return "Low"
    return "Unknown"

def recommendation(label):
    recs = {
        "Suicidal": "🚨 Immediate professional help is required. Contact emergency services or a suicide helpline.",
        "Depression": "Consider therapy, routine building, and social support.",
        "Anxiety": "Practice mindfulness, breathing exercises, and stress reduction.",
        "Stress": "Reduce workload, take breaks, and prioritize rest.",
        "Bipolar": "Consult a psychiatrist for mood stabilization.",
        "Personality_disorder": "Long-term therapy and professional support are advised.",
        "Normal": "No significant mental health concerns detected. Maintain healthy habits and positive routines."
    }
    return recs.get(label, "Seek professional guidance if needed.")

def contains_suicide_keywords(text):
    keywords = [
        "kill myself", "end my life", "want to die", "suicide",
        "better off dead", "no reason to live", "end it all"
    ]
    t = text.lower()
    return any(k in t for k in keywords)

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Invalid input"}), 400

    text = data["text"].strip()
    if len(text) < 5:
        return jsonify({"error": "Text too short"}), 400

    cleaned = clean_text(text)
    sentiment = sentiment_info(text)

    inputs = tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    label = label_classes[pred_idx]
    confidence = float(probs[pred_idx])

    # Get probabilities of illness classes
    mental_illness_indices = [i for i, lbl in enumerate(label_classes)
                             if lbl not in ["Normal"]]
    mental_illness_prob = np.sum([probs[i] for i in mental_illness_indices])
    normal_idx = label_classes.index("Normal")
    normal_prob = probs[normal_idx]

    # 🎯 CONSERVATIVE CLASSIFICATION BASED ON DATASET
    # Dataset has: Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality_disorder
    # Trust the model's predictions but add minimal sentiment-based adjustments

    # Only boost Normal for strongly positive sentiment with very weak illness signals
    if sentiment > 0.3 and mental_illness_prob < 0.3 and normal_prob > 0.4:
        label = "Normal"
        confidence = max(confidence, 0.85)

    # For borderline cases where Normal is close to winning, let the model decide
    elif label != "Normal" and normal_prob > 0.25 and normal_prob > mental_illness_prob * 0.9:
        # Only switch if Normal is very close to the predicted class
        if abs(normal_prob - confidence) < 0.15:
            label = "Normal"
            confidence = normal_prob
    
    # For Normal predictions, verify they're solid
    if label == "Normal":
        mental_illness_indices = [i for i, lbl in enumerate(label_classes) 
                                 if lbl not in ["Normal"]]
        mental_illness_prob = np.sum([probs[i] for i in mental_illness_indices])
        
        # Boost confidence if mental illness signals are very weak
        if mental_illness_prob < 0.30:
            confidence = max(confidence, 0.70)
        elif mental_illness_prob > 0.60 and confidence < 0.30:
            # High illness signals but Normal won - reconsider
            alt_idx = np.argmax([probs[i] for i in mental_illness_indices])
            label = label_classes[mental_illness_indices[alt_idx]]
            confidence = probs[mental_illness_indices[alt_idx]]

    # 🔐 SAFETY BOOST (SUICIDAL DETECTION)
    if contains_suicide_keywords(text) and label != "Suicidal":
        suicide_idx = label_classes.index("Suicidal")
        suicide_prob = probs[suicide_idx]
        if suicide_prob > 0.25:   # threshold
            label = "Suicidal"
            confidence = suicide_prob

    risk = risk_from_label(label)
    confidence_pct = round(confidence * 100, 1)

    return jsonify({
        "category": label,
        "risk": risk,
        "confidence": confidence_pct,
        "sentiment_score": sentiment,
        "recommendation": recommendation(label),
        "probabilities": {
            label_classes[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(label_classes))
        }
    })

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    print("🚀 Server running at http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)
