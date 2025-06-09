import streamlit as st
import pickle
import numpy as np
from textblob import TextBlob

# Load model and vectorizer
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    tfidf = pickle.load(vec_file)

# Title
st.title("Depression Tweet Classifier 💬")
st.write("Enter a tweet below to find out if it might be depressive.")

# Input box
user_input = st.text_area("Tweet text", placeholder="Type your tweet here...")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet before classifying.")
    else:
        # Vectorize the text
        tfidf_vector = tfidf.transform([user_input]).toarray()

        # Get sentiment score using TextBlob
        sentiment_score = TextBlob(user_input).sentiment.polarity
        sentiment_score = np.array([[sentiment_score]])

        # Combine features
        input_features = np.hstack((tfidf_vector, sentiment_score))

        # Predict
        prediction = model.predict(input_features)[0]
        label = "Depressive 😔" if prediction == 1 else "Not Depressive 🙂"

        st.subheader("Prediction:")
        st.success(label)
