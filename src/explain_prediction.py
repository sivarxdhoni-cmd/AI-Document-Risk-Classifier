import joblib


MODEL_PATH = "models/risk_classifier.joblib"


def explain_prediction(text):

    # Load trained pipeline
    model = joblib.load(MODEL_PATH)

    # Extract TF-IDF vectorizer and classifier
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    # Transform document
    text_vector = vectorizer.transform([text])

    # Prediction
    prediction = model.predict([text])[0]

    # Confidence
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    predicted_index = list(classes).index(prediction)
    confidence = probabilities[predicted_index]

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Get classifier coefficients
    class_index = list(classes).index(prediction)
    coefficients = classifier.coef_[class_index]

    # Get TF-IDF values
    tfidf_values = text_vector.toarray()[0]

    # Calculate contribution of each term
    contributions = tfidf_values * coefficients

    # Select terms that contributed positively
    important_terms = []

    for index, contribution in enumerate(contributions):
        if contribution > 0:
            important_terms.append(
                (feature_names[index], contribution)
            )

    # Sort by contribution
    important_terms.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "important_terms": important_terms[:10],
    }


def print_explanation(result):

    print("\n===== EXPLANATION =====")

    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']:.2%}")

    print("\nImportant terms:")

    if not result["important_terms"]:
        print("No strong contributing terms found.")
        return

    for term, contribution in result["important_terms"]:
        print(
            f"  {term:<25} "
            f"contribution = {contribution:.4f}"
        )


if __name__ == "__main__":

    sample_text = """
    Customer personal information including identity documents
    and financial details must be protected from unauthorized access.
    """

    result = explain_prediction(sample_text)

    print_explanation(result)