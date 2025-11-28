import os
import re
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping


class SentimentModelTrainer:
    """
    Trainer for a sentiment analysis model using a BiLSTM–CNN architecture.

    This class loads and parses review files, cleans and tokenises the text,
    builds a neural network model, and trains and evaluates it on a labelled
    dataset of reviews.

    Attributes:
        dataset_path (str): Base directory containing the review categories.
        model_dir (str): Directory where the trained model and tokenizer
            will be saved.
        MAX_WORDS (int): Maximum vocabulary size for the tokenizer.
        MAX_LEN (int): Maximum sequence length for padded input sequences.
        tokenizer (Tokenizer | None): Fitted Keras tokenizer instance.
        model (Sequential | None): Compiled Keras model instance.
    """

    def __init__(self, dataset_path="dataset", model_dir="model",
                 max_words=15000, max_len=300):
        """
        Initialise the trainer with dataset and model configuration.

        Args:
            dataset_path (str): Path to the root dataset directory.
            model_dir (str): Directory where the model and tokenizer will be stored.
            max_words (int): Maximum number of words to keep in the vocabulary.
            max_len (int): Maximum length of each input sequence after padding.
        """
        self.dataset_path = dataset_path
        self.model_dir = model_dir
        self.MAX_WORDS = max_words
        self.MAX_LEN = max_len

        # Ensure that the output directory exists before training.
        os.makedirs(self.model_dir, exist_ok=True)

        self.tokenizer = None
        self.model = None

    # ==========================================================
    # 1. Parsing .review files (custom XML-like format)
    # ==========================================================

    def parse_review_file(self, filepath):
        """
        Parse a .review file and extract all review texts.

        The file is expected to contain multiple <review> blocks with
        <review_text> tags. Only non-empty review texts are returned.

        Args:
            filepath (str): Path to the .review file.

        Returns:
            list[str]: List of review text strings extracted from the file.
        """
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Extract all <review> ... </review> blocks from the file.
        blocks = re.findall(r"<review>(.*?)</review>", content, re.DOTALL)

        reviews = []
        for block in blocks:
            match = re.search(r"<review_text>(.*?)</review_text>", block, re.DOTALL)
            if match:
                text = match.group(1).strip()
                if text:
                    reviews.append(text)

        return reviews

    # ==========================================================
    # 2. Loading the full dataset
    # ==========================================================

    def load_all_reviews(self):
        """
        Load all reviews and their labels from the dataset directory.

        The dataset directory is expected to contain one subdirectory per
        category. Each category directory must contain two files:
        'positive.review' and 'negative.review'. Positive reviews are
        labelled with 1 and negative reviews with 0.

        Returns:
            tuple[list[str], list[int]]: A tuple containing:
                - texts: List of raw review texts.
                - labels: List of integer sentiment labels (1 for positive,
                  0 for negative).
        """
        categories = os.listdir(self.dataset_path)

        texts = []
        labels = []

        for cat in categories:
            cat_path = os.path.join(self.dataset_path, cat)
            pos_path = os.path.join(cat_path, "positive.review")
            neg_path = os.path.join(cat_path, "negative.review")

            print(f"📂 Loading category: {cat}")

            # Load and label positive reviews.
            pos_reviews = self.parse_review_file(pos_path)
            texts.extend(pos_reviews)
            labels.extend([1] * len(pos_reviews))

            # Load and label negative reviews.
            neg_reviews = self.parse_review_file(neg_path)
            texts.extend(neg_reviews)
            labels.extend([0] * len(neg_reviews))

        print(f"Total reviews loaded: {len(texts)}")
        return texts, labels

    # ==========================================================
    # 3. Text cleaning and basic filtering
    # ==========================================================

    def clean_text(self, text):
        """
        Clean a single review string.

        The cleaning steps include:
            - Converting to lower case.
            - Removing HTML-like tags.
            - Removing URLs.
            - Removing non-alphanumeric characters.
            - Normalising whitespace.

        Args:
            text (str): Raw review text.

        Returns:
            str: Cleaned review text.
        """
        text = text.lower()
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def remove_outliers(self, texts, labels, min_words=5):
        """
        Remove very short reviews that may be noisy or uninformative.

        Any review with fewer than `min_words` tokens is discarded along
        with its corresponding label.

        Args:
            texts (list[str]): List of cleaned review texts.
            labels (list[int]): List of sentiment labels aligned with `texts`.
            min_words (int): Minimum number of words required to keep a review.

        Returns:
            tuple[list[str], list[int]]: A tuple containing:
                - cleaned_texts: Filtered list of review texts.
                - cleaned_labels: Corresponding list of labels.
        """
        cleaned_texts = []
        cleaned_labels = []

        for t, l in zip(texts, labels):
            if len(t.split()) >= min_words:
                cleaned_texts.append(t)
                cleaned_labels.append(l)

        print(f"Reviews after outlier removal: {len(cleaned_texts)}")
        return cleaned_texts, cleaned_labels

    # ==========================================================
    # 4. Tokenisation and padding
    # ==========================================================

    def tokenize_and_pad(self, texts):
        """
        Tokenise the text corpus and pad sequences to a fixed length.

        A Keras Tokenizer is fitted on the input texts, and each text is
        converted to a sequence of integer word indices. These sequences
        are then padded (or truncated) to `self.MAX_LEN` tokens.

        Args:
            texts (list[str]): List of cleaned review texts.

        Returns:
            np.ndarray: 2D array of shape (n_samples, MAX_LEN) containing
            the padded integer sequences.
        """
        tokenizer = Tokenizer(num_words=self.MAX_WORDS, oov_token="<UNK>")
        tokenizer.fit_on_texts(texts)

        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=self.MAX_LEN,
            padding="post",
            truncating="post"
        )

        # Store the fitted tokenizer for later use (e.g., during inference).
        self.tokenizer = tokenizer
        return padded

    # ==========================================================
    # 5. Building the BiLSTM + CNN model
    # ==========================================================

    def build_model(self):
        """
        Build and compile the sentiment analysis model.

        The model uses:
            - An embedding layer.
            - A bidirectional LSTM with sequence output.
            - A 1D convolutional layer followed by global max pooling.
            - Fully connected layers with dropout.
            - A final sigmoid unit for binary classification.

        Returns:
            Sequential: Compiled Keras Sequential model.
        """
        model = Sequential([
            Embedding(self.MAX_WORDS, 128),

            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.5),

            Conv1D(64, kernel_size=3, activation="relu"),
            GlobalMaxPooling1D(),

            Dense(64, activation="relu"),
            Dropout(0.5),

            Dense(1, activation="sigmoid")
        ])

        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        self.model = model
        return model

    # ==========================================================
    # 6. Full training pipeline (model generation)
    # ==========================================================

    def generate_model(self):
        """
        Run the full training pipeline and save the trained model and tokenizer.

        This method:
            1. Loads all reviews and labels from disk.
            2. Cleans and filters the text.
            3. Tokenises and pads the text sequences.
            4. Splits the data into training, validation, and test sets.
            5. Builds and trains the model with early stopping.
            6. Evaluates the model on the test set.
            7. Saves the trained model and tokenizer to disk.

        Returns:
            bool: True if the model and tokenizer were saved successfully;
            False if an error occurred during saving.
        """
        print("\n===== STARTING MODEL TRAINING PIPELINE =====\n")

        # 1. Load raw texts and labels from all categories.
        texts, labels = self.load_all_reviews()

        # 2. Clean text and remove very short reviews.
        texts = [self.clean_text(t) for t in texts]
        texts, labels = self.remove_outliers(texts, labels)

        labels = np.array(labels)

        # 3. Convert text to padded integer sequences.
        padded = self.tokenize_and_pad(texts)

        # 4. Split into training, validation, and test subsets.
        X_train, X_temp, y_train, y_temp = train_test_split(
            padded, labels, test_size=0.30, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42
        )

        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # 5. Build the neural network model.
        model = self.build_model()

        # 6. Train with early stopping based on validation loss.
        es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=64,
            callbacks=[es]
        )

        # 7. Evaluate the final model on the held-out test set.
        loss, acc = model.evaluate(X_test, y_test)
        print(f"\n🎉 Final Test Accuracy: {acc:.4f}")

        try:
            # 8. Save the trained model.
            model.save(f"{self.model_dir}/sentiment_model.keras")
            print(f"💾 Model saved at: {self.model_dir}/sentiment_model.keras")

            # 9. Save the fitted tokenizer for later inference.
            with open(f"{self.model_dir}/tokenizer.pkl", "wb") as f:
                pickle.dump(self.tokenizer, f)

            print(f"💾 Tokenizer saved at: {self.model_dir}/tokenizer.pkl")
            print("\n===== TRAINING COMPLETE =====\n")
            return True  # ---------- SUCCESS
        except Exception as e:
            print("❌ ERROR while generating the model:")
            print(e)
            return False  # ---------- FAILED