"""
Flask web application for interactive sentiment analysis of product reviews.

This module:
    - Loads a pre-trained sentiment analysis model and its corresponding tokenizer.
    - Exposes a simple web interface where users can type or select an example review.
    - Returns a sentiment label (Positive/Negative) and a probability score, along with
      a visual progress bar and an interpretation table.
"""

# app.py
from flask import Flask, request, render_template_string
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

# ---------------------------------------------------------------------------
# Model and tokenizer configuration
# ---------------------------------------------------------------------------

# Directory and file paths for the trained model and tokenizer.
MODEL_DIR = "./npl/model"
MODEL_PATH = f"{MODEL_DIR}/sentiment_model.keras"
TOKENIZER_PATH = f"{MODEL_DIR}/tokenizer.pkl"

# Examples of reviews for quick testing via the sidebar in the web interface.
# These are illustrative and do not affect the model itself.
SAMPLE_REVIEWS = [
    "This product is amazing, it exceeded all my expectations.",
    "This product is an absolute disaster. It stopped working properly within the first 24 hours, and the few times it did work, it performed terribly. The materials feel cheap, the system glitches constantly, and the instructions are confusing and poorly written. When I reached out for help, the customer service team was rude and completely unhelpful. I honestly regret spending even a single dollar on this. Save yourself the frustration and look for something else—this has been one of the worst shopping experiences I’ve ever had.",
    "I recently purchased this product after reading several recommendations, and overall, I am quite satisfied with the performance. The build quality is solid, the features are intuitive, and it works exactly as advertised. I especially appreciate how easy it was to set up, even for someone who isn’t very tech-savvy. It’s not perfect—battery life could be slightly better, and the packaging felt a bit cheap—but these are minor issues compared to the value it provides. I would definitely recommend it to anyone looking for something reliable at a reasonable price.",
    "Very disappointed, it stopped working after a week.",
    "Excellent value for money, I highly recommend it.",
    "The packaging was damaged but the product was fine.",
    "I had high expectations for this product, but unfortunately it didn’t live up to what was promised. While it looks nice on the outside, the performance is inconsistent and sometimes frustrating. After a few days of use, it began to lag and occasionally stopped working altogether. I tried contacting customer support but only received generic responses that didn’t solve the issue. For the price, I honestly expected something more durable and dependable. It’s not the worst thing I’ve bought, but I wouldn’t purchase it again.",
    "I love it! I use it every single day.",
    "The instructions were confusing and not helpful.",
    "This is hands down one of the best products I’ve ever purchased. From the moment I opened the box, everything felt premium—the materials, the design, and especially the performance. It runs smoothly, delivers exactly what it promises, and even exceeds expectations in some areas. Customer service was also outstanding; they responded quickly and were extremely helpful. I’ve been using it daily without any issues, and it has genuinely improved my routine. I highly recommend it to anyone looking for quality and long-term value."
]

# ---------------------------------------------------------------------------
# Model and tokenizer loading (performed once at application startup)
# ---------------------------------------------------------------------------

print("Loading model and tokenizer...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load tokenizer (pickle file contains only the fitted tokenizer).
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Maximum sequence length, which must match the value used during training.
MAXLEN = 300

# ---------------------------------------------------------------------------
# Flask application setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

# HTML template for the web interface. This is rendered using Flask's
# `render_template_string` and includes:
#     - A text area for entering or editing a review.
#     - A result panel with emoji feedback and a progress bar.
#     - A table explaining how to interpret the score.
#     - A sidebar with random example reviews.
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
        margin-left: 20px;
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
    <p>Type or select a random Amazon review and click "Analyse Sentiment".</p>

    <div class="layout">
      <div class="main">
        <form method="post">
          <textarea id="review-textarea" 
                    name="review" 
                    placeholder="Write your review here...">{{ text or "" }}</textarea>

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
              <br><br>
          <button type="submit">Analyse Sentiment</button>
        </form>



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


      </div>

      <div class="sidebar">
        <h2>Random Amazon Reviews</h2>
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


# ---------------------------------------------------------------------------
# Pre-processing and prediction utilities
# ---------------------------------------------------------------------------


def preprocess_text(text: str):
    """
    Convert a raw review string into a padded integer sequence.

    The function assumes that `tokenizer` has already been fitted on the
    training corpus and that `MAXLEN` matches the value used during model
    training.

    Args:
        text (str): Raw review text to be processed.

    Returns:
        np.ndarray: A 2D array of shape (1, MAXLEN) representing the
        padded sequence for the input text.
    """
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")
    return padded


def predict_sentiment(text: str):
    """
    Generate a sentiment prediction for the given review text.

    The function:
        1. Strips whitespace and checks that the input is not empty.
        2. Pre-processes the text into a padded integer sequence.
        3. Uses the loaded model to predict a sentiment probability.
        4. Maps the probability to a label ("Positive" or "Negative").

    Args:
        text (str): Raw review text provided by the user.

    Returns:
        tuple[str | None, float | None]:
            - sentiment (str or None): "Positive" or "Negative" if prediction
              is successful, otherwise None for empty input.
            - prob (float or None): Predicted probability of the positive class
              (between 0 and 1), or None if the input was empty.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        # No valid text to analyse.
        return None, None

    padded = preprocess_text(clean_text)
    pred = model.predict(padded, verbose=0)[0][0]
    prob = float(pred)
    sentiment = "Positive" if prob >= 0.5 else "Negative"
    return sentiment, prob


def get_random_reviews(n: int = 5):
    """
    Select a small set of random example reviews for the sidebar.

    Args:
        n (int): Number of reviews to return. The value is capped at the
            length of SAMPLE_REVIEWS to avoid errors.

    Returns:
        list[str]: List of randomly selected review strings. Returns an
        empty list if SAMPLE_REVIEWS is empty.
    """
    if not SAMPLE_REVIEWS:
        return []
    n = min(n, len(SAMPLE_REVIEWS))
    return random.sample(SAMPLE_REVIEWS, n)


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route for the sentiment analysis web interface.

    Handles both GET and POST requests:
        - GET: Renders the form with an empty text area and random example reviews.
        - POST: Reads the user's review from the form, computes sentiment,
          and re-renders the page with the result and updated examples.

    Returns:
        str: Rendered HTML page as a string.
    """
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
        random_reviews=random_reviews,
    )


# ---------------------------------------------------------------------------
# Application entry point helper
# ---------------------------------------------------------------------------

def start_flask_app():
    """
    Start the Flask development server.

    The server is configured to:
        - Listen on all available network interfaces (0.0.0.0).
        - Use port 5001.
        - Run with debug mode disabled (suitable for simple deployment or
          classroom demonstrations, but not for production).
    """
    app.run(host="0.0.0.0", port=5001, debug=False)