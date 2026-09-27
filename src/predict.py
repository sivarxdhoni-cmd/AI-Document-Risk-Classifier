import joblib


MODEL_PATH = "models/risk_classifier.joblib"

# Documents below this confidence will be sent for manual review
CONFIDENCE_THRESHOLD = 0.70


def predict_document(text):
    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Predict class
    prediction = model.predict([text])[0]

    # Get probability for each class
    probabilities = model.predict_proba([text])[0]

    # Get class names
    classes = model.classes_

    # Find confidence of predicted class
    predicted_index = list(classes).index(prediction)
    confidence = probabilities[predicted_index]

    # Decide final status
    if confidence >= CONFIDENCE_THRESHOLD:
        status = "AUTO_CLASSIFIED"
    else:
        status = "MANUAL_REVIEW"

    return {
        "prediction": prediction,
        "confidence": confidence,
        "status": status,
        "probabilities": dict(zip(classes, probabilities)),
    }


def print_result(result):

    print("\n===== DOCUMENT ANALYSIS =====")

    print(f"Predicted Risk : {result['prediction']}")
    print(f"Confidence     : {result['confidence']:.2%}")
    print(f"Status         : {result['status']}")

    print("\nClass Probabilities:")

    for label, probability in result["probabilities"].items():
        print(f"  {label:<8}: {probability:.2%}")


if __name__ == "__main__":

    sample_text = """
    Customer personal information including identity documents
    and financial details must be protected from unauthorized access.
    """

    result = predict_document(sample_text)

    print_result(result)