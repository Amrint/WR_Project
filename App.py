import streamlit as st
import pickle
import numpy as np

# Load the trained model and vectorizer
with open('best_model.pkl', 'rb') as model_file, open('best_vectorizer.pkl', 'rb') as vec_file:
    model = pickle.load(model_file)
    vectorizer = pickle.load(vec_file)

# Title of the Streamlit app
st.title("Sentiment Analysis Web App")
st.write("Predict the sentiment of your text: Positive, Negative, or Neutral.")

# Input text box
user_input = st.text_area("Enter your text here:")

# Prediction
if st.button("Predict Sentiment"):
    if user_input.strip():
        # Preprocess and vectorize input
        input_vectorized = vectorizer.transform([user_input])
        
        # Predict sentiment
        prediction = model.predict(input_vectorized)
        prediction_proba = model.predict_proba(input_vectorized)

        # Display the result
        st.subheader("Prediction:")
        st.write(f"The sentiment is **{prediction[0].capitalize()}**.")

        st.subheader("Confidence:")
        for sentiment, prob in zip(model.classes_, prediction_proba[0]):
            st.write(f"{sentiment.capitalize()}: {prob * 100:.2f}%")
    else:
        st.error("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit")
