# ==============================
# Mental Health Web App (Flask)
# Dataset: Kaggle Combined.csv
# ==============================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import re

# ML & NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Load Dataset
# -------------------------------
DATA_PATH = "data/Combined.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("❌ Dataset not found. Place Combined.csv inside data/ folder")

data = pd.read_csv(DATA_PATH)
data = data.dropna(subset=["statement", "status"])

X = data["statement"]
y = data["status"]

print("✅ Dataset loaded:", data.shape)

# -------------------------------
# Train ML Model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)  # improves confidence stability
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",   # Changed from liblinear to support multiclass classification
    
)
model.fit(X_train_vec, y_train)

print("✅ Model trained successfully")

# -------------------------------
# Label Mapping (DATASET EXACT)
# -------------------------------
LABEL_MAP = {
    "normal": "Normal",
    "anxiety": "Anxiety",
    "depression": "Depression",
    "stress": "Stress",
    "bipolar": "Bipolar",
    "ptsd": "PTSD",
    "suicidal": "Suicidal"
}

# -------------------------------
# Suicide Intent Keywords (SAFETY)
# -------------------------------
SUICIDE_KEYWORDS = [
    "want to die", "kill myself", "end my life",
    "suicide", "better off dead", "no reason to live",
    "wish i were dead", "end it all"
]

# -------------------------------
# Risk Mapping (DATASET-ALIGNED)
# -------------------------------
def map_risk(category):
    if category in ["Suicidal", "Bipolar", "PTSD"]:
        return "High"
    elif category in ["Depression", "Anxiety", "Stress"]:
        return "Medium"
    return "Low"

# -------------------------------
# Recommendation Engine
# -------------------------------
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
        return "Trauma-focused therapy and professional counseling are strongly recommended."

    if category == "Depression":
        return "Therapy, regular routine, physical activity, and social support can be beneficial."

    if category == "Anxiety":
        return "Mindfulness, breathing exercises, reduced caffeine intake, and professional guidance may help."

    if category == "Stress":
        return "Time management, adequate rest, and relaxation techniques are recommended."

    return "Maintain healthy habits and continue positive coping strategies."

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()
    user_text = data.get("text", "").strip()

    if not user_text:
        return jsonify({"error": "Empty input"}), 400

    # ---------------- ML Prediction ----------------
    user_vec = vectorizer.transform([user_text])

    raw_prediction = model.predict(user_vec)[0]

    # Confidence score (SAFE)
    proba = model.predict_proba(user_vec)[0]
    confidence = round(float(max(proba)) * 100, 1)

    category = LABEL_MAP.get(raw_prediction.lower(), raw_prediction)

    # ---------------- Sentiment Analysis ----------------
    blob = TextBlob(user_text)
    sentiment_score = round(blob.sentiment.polarity, 3)

    if sentiment_score > 0.2:
        sentiment_label = "Positive"
    elif sentiment_score < -0.2:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    # -------- SAFETY OVERRIDE --------
    lower_text = user_text.lower()

    if category == "Suicidal" or any(k in lower_text for k in SUICIDE_KEYWORDS):
        sentiment_label = "Negative"
        sentiment_score = -0.7

    # ---------------- Risk ----------------
    risk = map_risk(category)

    # ---------------- Recommendation ----------------
    recommendation = get_recommendation(category, risk)

    # ---------------- Response ----------------
    return jsonify({
        "category": category,
        "risk": risk,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "recommendation": recommendation,
        "word_count": len(user_text.split()),
        "confidence": confidence
    })

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)