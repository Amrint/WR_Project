import streamlit as st
import pickle
import pandas as pd

# Title and Description
st.title("Sentiment Analysis App")
st.write("Analyze sentiment as Positive, Negative, or Neutral.")

# Load Model and Vectorizer with Error Handling
@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open('best_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('best_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError as e:
        st.error("Model or vectorizer file not found. Please upload `best_model.pkl` and `best_vectorizer.pkl` to the same directory as `app.py`.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model or vectorizer: {str(e)}")
        st.stop()

# Load the resources
model, vectorizer = load_model_and_vectorizer()

# Input Text
text_input = st.text_area("Enter text to analyze sentiment:")

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocessing the input
        text_input_cleaned = " ".join(text_input.lower().split())  # Basic cleaning for this example
        
        # Transform the input
        try:
            text_vectorized = vectorizer.transform([text_input_cleaned])
            prediction = model.predict(text_vectorized)[0]
            st.success(f"Predicted Sentiment: **{prediction.capitalize()}**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
