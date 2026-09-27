import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


TRAIN_PATH = "data/train.csv"
MODEL_PATH = "models/risk_classifier.joblib"


def train_model():

    print("Loading training dataset...")

    df = pd.read_csv(TRAIN_PATH)

    X = df["text"]
    y = df["label"]

    print(f"Training documents: {len(df)}")
    print(f"Classes: {sorted(y.unique())}")

    # TF-IDF converts text into numerical features
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )

    # Logistic Regression performs classification
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    # Combine TF-IDF + classifier
    model = Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )

    print("\nTraining model...")

    model.fit(X, y)

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Save complete pipeline
    joblib.dump(model, MODEL_PATH)

    print("\nModel training completed successfully.")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()