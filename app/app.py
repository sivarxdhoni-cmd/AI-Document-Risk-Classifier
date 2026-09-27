import streamlit as st
import joblib
import fitz
from docx import Document
import io


# =========================
# CONFIGURATION
# =========================

MODEL_PATH = "models/risk_classifier.joblib"
CONFIDENCE_THRESHOLD = 0.70


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# =========================
# TEXT EXTRACTION
# =========================

def extract_pdf_text(file_bytes):
    text = ""

    pdf = fitz.open(stream=file_bytes, filetype="pdf")

    for page in pdf:
        text += page.get_text()

    pdf.close()

    return text


def extract_docx_text(file_bytes):
    document = Document(io.BytesIO(file_bytes))

    paragraphs = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs)


def extract_txt_text(file_bytes):
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_pdf_text(file_bytes)

    elif file_name.endswith(".docx"):
        return extract_docx_text(file_bytes)

    elif file_name.endswith(".txt"):
        return extract_txt_text(file_bytes)

    return ""


# =========================
# AI PREDICTION
# =========================

def predict_document(text):

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]

    classes = model.classes_

    predicted_index = list(classes).index(prediction)

    confidence = probabilities[predicted_index]

    if confidence >= CONFIDENCE_THRESHOLD:
        status = "AUTO_CLASSIFIED"
    else:
        status = "MANUAL_REVIEW"

    return prediction, confidence, status, probabilities, classes


# =========================
# EXPLAINABILITY
# =========================

def get_important_terms(text, prediction):

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    text_vector = vectorizer.transform([text])

    feature_names = vectorizer.get_feature_names_out()

    classes = classifier.classes_

    class_index = list(classes).index(prediction)

    coefficients = classifier.coef_[class_index]

    tfidf_values = text_vector.toarray()[0]

    contributions = tfidf_values * coefficients

    important_terms = []

    for index, contribution in enumerate(contributions):

        if contribution > 0:

            important_terms.append(
                (
                    feature_names[index],
                    contribution
                )
            )

    important_terms.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return important_terms[:10]


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="AI Document Risk Classifier",
    page_icon="📄",
    layout="wide"
)


st.title("📄 AI Document Risk & Compliance Classifier")

st.write(
    "Upload a business document to analyze its potential risk category."
)


uploaded_file = st.file_uploader(
    "Upload PDF, DOCX or TXT",
    type=["pdf", "docx", "txt"]
)


if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    # Extract document text
    text = extract_text(uploaded_file)

    if not text.strip():

        st.error(
            "No readable text was found in the uploaded document."
        )

    else:

        # Show extracted text
        with st.expander("📄 View Extracted Text"):

            st.text_area(
                "Document Text",
                text,
                height=250
            )

        # Analyze button
        if st.button(
            "🔍 Analyze Document",
            type="primary"
        ):

            prediction, confidence, status, probabilities, classes = (
                predict_document(text)
            )

            important_terms = get_important_terms(
                text,
                prediction
            )

            st.divider()

            st.subheader("🤖 AI Analysis")


            # Risk + confidence columns
            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Risk Category",
                    prediction.upper()
                )


            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2%}"
                )


            with col3:

                if status == "AUTO_CLASSIFIED":

                    st.success("AUTO CLASSIFIED")

                else:

                    st.warning("MANUAL REVIEW")


            # Manual review warning
            if status == "MANUAL_REVIEW":

                st.warning(
                    "⚠️ The model confidence is below "
                    f"{CONFIDENCE_THRESHOLD:.0%}. "
                    "Human review is recommended."
                )

            else:

                st.success(
                    "✅ Model confidence is above the "
                    "configured threshold."
                )


            # Probability chart
            st.subheader("📊 Risk Probabilities")

            probability_data = {}

            for label, probability in zip(
                classes,
                probabilities
            ):

                probability_data[
                    label.upper()
                ] = probability


            st.bar_chart(probability_data)


            # Explainability
            st.subheader("🔍 Important Terms")

            if important_terms:

                for term, contribution in important_terms:

                    st.write(
                        f"• **{term}** "
                        f"— contribution: "
                        f"{contribution:.4f}"
                    )

            else:

                st.info(
                    "No strong contributing terms found."
                )


            # Disclaimer
            st.divider()

            st.caption(
                "This system provides AI-assisted document "
                "risk screening and compliance triage. "
                "It is not a substitute for professional "
                "legal or compliance advice."
            )