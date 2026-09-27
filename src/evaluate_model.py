import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


TEST_PATH = "data/test.csv"
MODEL_PATH = "models/risk_classifier.joblib"


def evaluate_model():

    print("Loading test dataset...")
    test_df = pd.read_csv(TEST_PATH)

    print(f"Test documents: {len(test_df)}")

    # Load trained model
    model = joblib.load(MODEL_PATH)

    X_test = test_df["text"]
    y_test = test_df["label"]

    # Make predictions
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy percentage: {accuracy * 100:.2f}%")

    # Classification report
    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_test,
            predictions,
            labels=["low", "medium", "high"],
            zero_division=0,
        )
    )

    # Confusion matrix
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["low", "medium", "high"],
    )

    print("===== CONFUSION MATRIX =====")

    print("Rows = Actual")
    print("Columns = Predicted")
    print("\nLabels: [low, medium, high]")
    print(matrix)

    # Show individual predictions
    print("\n===== INDIVIDUAL PREDICTIONS =====")

    for index, prediction in enumerate(predictions):

        actual = y_test.iloc[index]

        print(
            f"Document {test_df['document_id'].iloc[index]} "
            f"| Actual: {actual} "
            f"| Predicted: {prediction}"
        )


if __name__ == "__main__":
    evaluate_model()