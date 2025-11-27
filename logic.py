# logic.py
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

MODEL_DIR = "model"

# Load model
model = load_model(os.path.join(MODEL_DIR, "sentiment_model.keras"))

# Load tokenizer + maxlen
with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
    data = pickle.load(f)
tokenizer = data["tokenizer"]
MAXLEN = data["maxlen"]

def analyse_sentiment(text):
    """Return (sentiment_label, confidence_score) for a review string."""
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")

    prob = float(model.predict(padded)[0][0])
    label = "Positive" if prob >= 0.5 else "Negative"

    return label, prob
