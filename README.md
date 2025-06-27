# Mental_Health_Story_Analyzer
This project is an AI-powered NLP pipeline that analyzes user-written stories to detect emotions.
A powerful NLP-based tool designed to **analyze personal mental health stories**, detect the **underlying emotion**, identify **cognitive distortions**, and provide **supportive advice**. This project aims to combine **deep learning**, **rule-based logic**, and a **user-friendly Streamlit interface** to offer empathetic feedback and promote mental well-being.

## 💡 Key Features

- 🧾 **Emotion Detection** — Uses a trained ANN model to classify emotions like joy, sadness, anger, etc.
- 🧠 **Cognitive Distortion Detection** — Detects thought patterns (e.g., overgeneralization, catastrophizing) via rule-based or ML methods.
- 💬 **Advice Generation** — Offers constructive and supportive advice based on detected emotions.
- 🎛️ **Interactive UI** — Streamlit-based interface to enter, analyze, and visualize results in real time.

---

## 📂 Project Structure

mental-health-analyzer/
├── app.py # Streamlit app (Frontend)
├── Mental_Health_Story_Analyzer_model_pipeline.py # Core NLP + ML pipeline (Backend)
├── emotion_model.h5 # Trained emotion classification model
├── advice_model.h5 # Trained advice classification model
├── tfidf_vectorizer.pkl # TF-IDF vectorizer for text features
├── emotion_encoder.pkl # Label encoder for emotion labels
├── advice_encoder.pkl # Label encoder for advice labels
├── requirements.txt # List of required libraries
├── README.md # Project documentation
