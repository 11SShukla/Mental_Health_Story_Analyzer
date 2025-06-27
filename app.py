# app.py

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models and encoders
emotion_model = load_model("emotion_model.h5")
advice_model = load_model("advice_model.h5")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
emotion_encoder = joblib.load("emotion_encoder.pkl")
advice_encoder = joblib.load("advice_encoder.pkl")

# Title
st.title("🧠 Mental Health Story Analyzer")
st.markdown("Analyze your thoughts and get emotional insight, detect distortions, and receive supportive advice.")

# Text input
user_input = st.text_area("Enter your mental health story here 👇", height=200)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a valid story.")
    else:
        # Vectorize text
        X = vectorizer.transform([user_input])

        # Emotion prediction
        emotion_probs = emotion_model.predict(X)
        emotion_index = np.argmax(emotion_probs)
        emotion_label = emotion_encoder.inverse_transform([emotion_index])[0]

        # Advice prediction
        advice_probs = advice_model.predict(X)
        advice_index = np.argmax(advice_probs)
        advice_label = advice_encoder.inverse_transform([advice_index])[0]

        # Rule-based cognitive distortion detection (simplified)
        def detect_distortion(text):
            text = text.lower()
            if "always" in text or "never" in text:
                return "Overgeneralization"
            elif "everyone hates me" in text:
                return "Personalization"
            elif "i'm a failure" in text:
                return "Labeling"
            elif "what if" in text:
                return "Catastrophizing"
            return "None detected"

        distortion = detect_distortion(user_input)

        # Display results
        st.subheader("🔍 Analysis Results")
        st.write(f"**Detected Emotion:** `{emotion_label}`")
        st.write(f"**Cognitive Distortion:** `{distortion}`")
        st.write(f"**Supportive Advice:** `{advice_label}`")
