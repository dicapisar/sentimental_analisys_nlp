# app.py
from flask import Flask, request, render_template_string
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

MODEL_DIR = "model"
MODEL_PATH = f"{MODEL_DIR}/sentiment_model.keras"
TOKENIZER_PATH = f"{MODEL_DIR}/tokenizer.pkl"

# Examples of evaluations (we can later replace these with actual evaluations of the dataset)
SAMPLE_REVIEWS = [
    "This product is amazing, it exceeded all my expectations.",
    "Terrible quality, I would not buy this again.",
    "The item arrived on time and works as described.",
    "Very disappointed, it stopped working after a week.",
    "Excellent value for money, I highly recommend it.",
    "The packaging was damaged but the product was fine.",
    "Not worth the price, I expected something better.",
    "I love it! I use it every single day.",
    "The instructions were confusing and not helpful.",
    "Great customer service and fast delivery."
]

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
      body { 
        font-family: Arial, sans-serif; 
        max-width: 1200px; 
        margin: 40px auto; 
      }
      .layout {
        display: flex;
        gap: 20px;
        align-items: flex-start;
      }

      .main {
        flex: 2;
      }

      .sidebar {
        flex: 1;
        border: 1px solid #ccc;
        padding: 10px;
        border-radius: 4px;
        background: #f9f9f9;
        max-height: 500px;
        overflow-y: auto;
      }
      .sidebar h2 {
        margin-top: 0;
        font-size: 18px;
      }
      .sidebar p {
        font-size: 13px;
        color: #555;
      }
      .sidebar ul {
        list-style: none;
        padding-left: 0;
      }
      .sidebar li {
        margin-bottom: 8px;
      }
      .sidebar button {
        width: 100%;
        text-align: left;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 6px 8px;
        font-size: 13px;
        background: #ffffff;
        cursor: pointer;
      }
      .sidebar button:hover {
        background: #eef5ff;
      }

      textarea { 
        width: 100%; 
        height: 150px; 
      }
      .result { 
        margin-top: 20px; 
        padding: 10px; 
        border-radius: 4px; 
      }
      .positive { 
        background: #e0ffe0; 
        border: 1px solid #4caf50; 
      }
      .negative { 
        background: #ffe0e0; 
        border: 1px solid #f44336; 
      }
    </style>
    <script>
      function selectReview(text) {
        const textarea = document.getElementById("review-textarea");
        if (textarea) {
          textarea.value = text;
          textarea.focus();
        }
      }
    </script>
  </head>
  <body>
    <h1>Amazon Review Sentiment Checker</h1>
    <p>Type or paste an Amazon-style review and click "Analyse Sentiment".</p>

    <div class="layout">
      <div class="main">
        <form method="post">
          <textarea id="review-textarea" 
                    name="review" 
                    placeholder="Write your review here...">{{ text or "" }}</textarea>
          <br><br>
          <button type="submit">Analyse Sentiment</button>
        </form>

        {% if sentiment %}
          <div class="result {{ sentiment|lower }}">
            <strong>Prediction:</strong> {{ sentiment }}<br>
            <strong>Confidence:</strong> {{ prob|round(3) }}
          </div>
        {% endif %}
      </div>

      <div class="sidebar">
        <h2>Random Reviews</h2>
        <p>Select one review to analyse it:</p>
        <ul>
          {% for r in random_reviews %}
          <li>
            <button type="button"
                    onclick="selectReview(this.dataset.review)"
                    data-review="{{ r|e }}">
              {{ r }}
            </button>
          </li>
          {% endfor %}
        </ul>
      </div>
    </div>
  </body>
</html>
"""


def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")
    return padded

def predict_sentiment(text: str):
    """
    Core prediction logic: - Cleans the text - Validates that it is not empty - Calls the model - Returns (sentiment_label, prob) or (None, None)
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return None, None

    padded = preprocess_text(clean_text)
    pred = model.predict(padded, verbose=0)[0][0]
    prob = float(pred)
    sentiment = "Positive" if prob >= 0.5 else "Negative"
    return sentiment, prob


def get_random_reviews(n: int = 5):
    """Returns n random reviews from SAMPLE_REVIEWS."""
    if not SAMPLE_REVIEWS:
        return []
    n = min(n, len(SAMPLE_REVIEWS))
    return random.sample(SAMPLE_REVIEWS, n)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    prob = None
    text = ""

    if request.method == "POST":
        text = request.form.get("review", "")
        sentiment, prob = predict_sentiment(text)

    random_reviews = get_random_reviews(5)

    return render_template_string(
        HTML_TEMPLATE,
        sentiment=sentiment,
        prob=prob,
        text=text,
        random_reviews = random_reviews,
    )


if __name__ == "__main__":
    # Debug server for local development
    app.run(host="0.0.0.0", port=5001, debug=True)

