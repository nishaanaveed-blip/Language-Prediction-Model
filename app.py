import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("language_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI setup
st.set_page_config(page_title="🌍 Language Detection App", layout="centered")

st.markdown("<h1 style='text-align:center; color:#2563EB;'>🌍 Language Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Detect the language of any text instantly using Naive Bayes & Scikit-learn</p>", unsafe_allow_html=True)

# Input area
user_input = st.text_area("Enter text to detect language:", height=150, placeholder="Type or paste any sentence here...")

if st.button("Detect Language"):
    if user_input.strip():
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)[0]

        # Confidence scores
        probs = model.predict_proba(transformed_text)[0]
        confidence = np.max(probs) * 100

        st.success(f"🗣 Detected Language: **{prediction}**")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter some text first!")
