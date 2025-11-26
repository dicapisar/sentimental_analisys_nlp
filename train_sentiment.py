import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
#  CONFIG
# -----------------------------
MAX_LEN = 400           # Increased sequence length
POS_THRESHOLD = 0.55    # More realistic threshold for positive sentiment
NEG_THRESHOLD = 0.45    # Below this → negative
# Between 0.45–0.55 → neutral


# -----------------------------
# LOAD MODEL & TOKENIZER
# -----------------------------
print("Loading model and tokenizer...")

MODEL_PATH = "model/sentiment_model.keras"
TOKENIZER_PATH = "model/tokenizer.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found at: " + MODEL_PATH)

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError("Tokenizer file not found at: " + TOKENIZER_PATH)

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


# -----------------------------
#  FLASK APP
# -----------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    sentiment = None

    if request.method == "POST":
        text = request.form["review"]

        # Avoid empty input errors
        if len(text.strip()) < 3:
            sentiment = "Input too short"
            confidence = "-"
            return render_template("index.html", sentiment=sentiment, confidence=confidence)

        # Tokenize
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        # Predict
        prob = float(model.predict(padded)[0][0])

        # Sentiment classification logic
        if prob > POS_THRESHOLD:
            sentiment = "Positive"
            confidence = round(prob, 4)
        elif prob < NEG_THRESHOLD:
            sentiment = "Negative"
            confidence = round(1 - prob, 4)
        else:
            sentiment = "Neutral"
            confidence = round(abs(prob - 0.5) * 2, 4)

        return render_template("index.html",
                               sentiment=sentiment,
                               confidence=confidence)

    return render_template("index.html")


# -----------------------------
#  RUN
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
