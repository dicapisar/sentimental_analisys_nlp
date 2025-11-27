# app.py
from flask import Flask, request, render_template_string
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_DIR = "./npl/model"
MODEL_PATH = f"{MODEL_DIR}/sentiment_model.keras"
TOKENIZER_PATH = f"{MODEL_DIR}/tokenizer.pkl"

# Load model + tokenizer at startup (only once)
print("Loading model and tokenizer...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load tokenizer (picker contains ONLY tokenizer)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

MAXLEN = 300   # same as used during training

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <title>ISY503 - Sentiment Analysis (NLP)</title>
    <style>
      body { 
        font-family: Arial, sans-serif; 
        max-width: 700px; 
        margin: 40px auto; 
      }
      textarea { 
        width: 100%; 
        height: 150px; 
        border-radius: 5px;
        padding: 10px;
        font-size: 14px;
      }
      button {
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
      }
      .result { 
        margin-top: 20px; 
        padding: 15px; 
        border-radius: 8px; 
        font-size: 18px;
        display: flex;
        align-items: center;
        gap: 15px;
      }
      .positive { 
        background: #e0ffe0; 
        border: 1px solid #4caf50;
      }
      .negative { 
        background: #ffe0e0; 
        border: 1px solid #f44336; 
      }
      /* === Progress Bar === */
      .bar-container {
        width: 100%;
        background: #eee;
        border-radius: 8px;
        overflow: hidden;
        margin-top: 8px;
        height: 20px;
      }
      .bar {
        height: 100%;
        transition: width 0.5s;
      }
      .bar-positive {
        background: #4caf50;
      }
      .bar-negative {
        background: #f44336;
      }
      /* === Table === */
      table {
        width: 100%;
        margin-top: 40px;
        border-collapse: collapse;
      }
      table th, table td {
        border: 1px solid #ccc;
        padding: 10px;
        text-align: center;
      }
      table th {
        background: #f0f0f0;
      }
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
        <span style="font-size: 40px;">
          {% if sentiment == "Positive" %}
            😊
          {% else %}
            😠
          {% endif %}
        </span>

        <div style="flex-grow: 1;">
          <strong>{{ sentiment }}</strong><br>
          <span>Score: {{ prob|round(3) }}</span>

          <!-- Progress bar -->
          <div class="bar-container">
            <div 
              class="bar {% if sentiment == 'Positive' %}bar-positive{% else %}bar-negative{% endif %}" 
              style="width: {{ (prob * 100)|round(0) }}%;">
            </div>
          </div>
        </div>
      </div>
    {% endif %}

    <h2>How to interpret the score</h2>
    <table>
      <tr>
        <th>Score Range</th>
        <th>Meaning</th>
        <th>Sentiment</th>
      </tr>
      <tr>
        <td>0.00 – 0.40</td>
        <td>Strongly Negative</td>
        <td>😠 Negative</td>
      </tr>
      <tr>
        <td>0.40 – 0.50</td>
        <td>Slightly Negative</td>
        <td>😐 Negative</td>
      </tr>
      <tr>
        <td>0.50 – 0.60</td>
        <td>Slightly Positive</td>
        <td>🙂 Positive</td>
      </tr>
      <tr>
        <td>0.60 – 1.00</td>
        <td>Strongly Positive</td>
        <td>😊 Positive</td>
      </tr>
    </table>

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


def start_flask_app():
    app.run(host="0.0.0.0", port=5001, debug=False)