# AI Document Risk Classifier

![Deployment Status](https://img.shields.io/badge/Deployment-Live-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/framework-Streamlit-red)

Welcome to the **AI Document Risk Classifier** project! This repository contains an intelligent tool designed to assess and classify risks associated with different types of documents. Using machine learning, the system can parse text from various document formats and predict their risk levels effectively.

## 🚀 Live Demo

You can try out the live deployed application here:
**[Launch AI Document Risk Classifier](https://ai-document-risk-classifier-njccbwjhjfbn7siqelegcj.streamlit.app/)**

---

## 🎯 Overview

The **AI Document Risk Classifier** allows users to upload documents and uses a pre-trained Machine Learning model to evaluate the textual content, assigning a risk classification based on the extracted data. It is built using a modern, interactive web interface powered by Streamlit.

### Key Features
- **Document Parsing**: Extracts text from `.pdf`, `.docx`, and raw `.txt` files.
- **Risk Classification**: Utilizes a Scikit-Learn machine learning pipeline to classify document risk.
- **Data Visualization**: Provides insights and metrics using Matplotlib and Seaborn.
- **Interactive UI**: An intuitive, web-based interface built entirely in Python using Streamlit.

---

## 🛠️ Technology Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Data Processing**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`
- **Model Serialization**: `joblib`
- **Document Extraction**: `PyMuPDF` (PDFs), `python-docx` (Word Documents)
- **Data Visualization**: `matplotlib`, `seaborn`

---

## 📂 Project Structure

```text
AI-Document-Risk-Classifier/
│
├── app/                        # Streamlit web application files
│   └── app.py                  # Main application script
├── data/                       # Datasets used for training/testing
│   ├── risk_dataset.csv
│   ├── train.csv
│   └── test.csv
├── models/                     # Saved Machine Learning models
│   └── risk_classifier.joblib  # Pre-trained ML model
├── src/                        # Source code for data and model pipelines
│   ├── create_dataset.py
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── explain_prediction.py
├── notebooks/                  # Jupyter notebooks for exploration
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 💻 Local Installation & Setup

If you'd like to run this project on your local machine, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/sivarxdhoni-cmd/AI-Document-Risk-Classifier.git
cd AI-Document-Risk-Classifier
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
Launch the web interface locally.
```bash
streamlit run app/app.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## 🧠 Machine Learning Pipeline

1. **Dataset Creation & Preparation**: The raw data (`data/risk_dataset.csv`) is preprocessed, cleaned, and split into training and testing sets using the scripts in the `src/` directory.
2. **Model Training**: A classification model is trained on the dataset using `scikit-learn` (`src/train_model.py`).
3. **Evaluation**: The model's accuracy and metrics are evaluated (`src/evaluate_model.py`).
4. **Inference**: The trained model is serialized using `joblib` (`models/risk_classifier.joblib`) and loaded into the Streamlit app to make real-time predictions.

---

## 📝 License

This project is open-source and available for educational and developmental purposes.
