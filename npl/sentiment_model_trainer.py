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

    def __init__(self, dataset_path="dataset", model_dir="model",
                 max_words=15000, max_len=300):
        self.dataset_path = dataset_path
        self.model_dir = model_dir
        self.MAX_WORDS = max_words
        self.MAX_LEN = max_len

        os.makedirs(self.model_dir, exist_ok=True)

        self.tokenizer = None
        self.model = None

    # ==========================================================
    # 1. Parseo de .review (sin XML)
    # ==========================================================

    def parse_review_file(self, filepath):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Extraer bloques <review> ... </review>
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
    # 2. Cargar dataset completo
    # ==========================================================

    def load_all_reviews(self):
        categories = os.listdir(self.dataset_path)

        texts = []
        labels = []

        for cat in categories:
            cat_path = os.path.join(self.dataset_path, cat)
            pos_path = os.path.join(cat_path, "positive.review")
            neg_path = os.path.join(cat_path, "negative.review")

            print(f"📂 Loading category: {cat}")

            pos_reviews = self.parse_review_file(pos_path)
            texts.extend(pos_reviews)
            labels.extend([1] * len(pos_reviews))

            neg_reviews = self.parse_review_file(neg_path)
            texts.extend(neg_reviews)
            labels.extend([0] * len(neg_reviews))

        print(f"Total reviews loaded: {len(texts)}")
        return texts, labels

    # ==========================================================
    # 3. Limpieza del texto
    # ==========================================================

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def remove_outliers(self, texts, labels, min_words=5):
        cleaned_texts = []
        cleaned_labels = []

        for t, l in zip(texts, labels):
            if len(t.split()) >= min_words:
                cleaned_texts.append(t)
                cleaned_labels.append(l)

        print(f"Reviews after outlier removal: {len(cleaned_texts)}")
        return cleaned_texts, cleaned_labels

    # ==========================================================
    # 4. Tokenización + Padding
    # ==========================================================

    def tokenize_and_pad(self, texts):
        tokenizer = Tokenizer(num_words=self.MAX_WORDS, oov_token="<UNK>")
        tokenizer.fit_on_texts(texts)

        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=self.MAX_LEN,
            padding="post",
            truncating="post"
        )

        self.tokenizer = tokenizer
        return padded

    # ==========================================================
    # 5. Modelo mejorado (BiLSTM + CNN)
    # ==========================================================

    def build_model(self):
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
    # 6. Generar modelo completo (pipeline final)
    # ==========================================================

    def generate_model(self):
        print("\n===== STARTING MODEL TRAINING PIPELINE =====\n")

        # 1. Load dataset
        texts, labels = self.load_all_reviews()

        # 2. Clean
        texts = [self.clean_text(t) for t in texts]
        texts, labels = self.remove_outliers(texts, labels)

        labels = np.array(labels)

        # 3. Tokenize + Pad
        padded = self.tokenize_and_pad(texts)

        # 4. Split
        X_train, X_temp, y_train, y_temp = train_test_split(
            padded, labels, test_size=0.30, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42
        )

        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # 5. Model
        model = self.build_model()

        # 6. Training
        es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=64,
            callbacks=[es]
        )

        # 7. Test
        loss, acc = model.evaluate(X_test, y_test)
        print(f"\n🎉 Final Test Accuracy: {acc:.4f}")

        try:
            # 8. Save model
            model.save(f"{self.model_dir}/sentiment_model.keras")
            print(f"💾 Model saved at: {self.model_dir}/sentiment_model.keras")

            # 9. Save tokenizer
            with open(f"{self.model_dir}/tokenizer.pkl", "wb") as f:
                pickle.dump(self.tokenizer, f)

            print(f"💾 Tokenizer saved at: {self.model_dir}/tokenizer.pkl")
            print("\n===== TRAINING COMPLETE =====\n")
            return True  # ---------- ÉXITO
        except Exception as e:
            print("❌ ERROR while generating the model:")
            print(e)
            return False  # ---------- FALLÓ