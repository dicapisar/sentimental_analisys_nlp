# app.py
from flask import Flask, request, render_template_string
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_DIR = "model"
MODEL_PATH = f"{MODEL_DIR}/sentiment_model.keras"
TOKENIZER_PATH = f"{MODEL_DIR}/tokenizer.pkl"

# Load model + tokenizer at startup (only once)
print("Loading model and tokenizer...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tok_data = pickle.load(f)
tokenizer = tok_data["tokenizer"]
MAXLEN = tok_data["maxlen"]

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <title>ISY503 - Sentiment Analysis (NLP)</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; }
      textarea { width: 100%; height: 150px; }
      .result { margin-top: 20px; padding: 10px; border-radius: 4px; }
      .positive { background: #e0ffe0; border: 1px solid #4caf50; }
      .negative { background: #ffe0e0; border: 1px solid #f44336; }
    </style>
  </head>
  <body>
    <h1>Amazon Review Sentiment Checker</h1>
    <p>Type or paste an Amazon-style review and click "Analyse Sentiment".</p>

    <form method="post">
      <textarea name="review" placeholder="Write your review here...">{{ text or "" }}</textarea>
      <br><br>
      <button type="submit">Analyse Sentiment</button>
    </form>

    {% if sentiment %}
      <div class="result {{ sentiment|lower }}">
        <strong>Prediction:</strong> {{ sentiment }}<br>
        <strong>Confidence:</strong> {{ prob|round(3) }}
      </div>
    {% endif %}
  </body>
</html>
"""


def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")
    return padded


@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    prob = None
    text = ""

    if request.method == "POST":
        text = request.form.get("review", "")
        if text.strip():
            padded = preprocess_text(text)
            pred = model.predict(padded, verbose=0)[0][0]
            prob = float(pred)
            sentiment = "Positive" if pred >= 0.5 else "Negative"

    return render_template_string(
        HTML_TEMPLATE,
        sentiment=sentiment,
        prob=prob,
        text=text
    )


if __name__ == "__main__":
    # Debug server for local development
    app.run(host="0.0.0.0", port=5001, debug=True)

