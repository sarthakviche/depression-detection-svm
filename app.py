import streamlit as st
import pickle
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="🧠 Tweet Depression Classifier",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for aesthetics ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        border-radius: 8px;
        font-weight: bold;
        height: 3em;
        width: 100%;
    }
    .stTextArea textarea {
        background-color: #eef2fb;
        border-radius: 8px;
        font-size: 1.1em;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2575fc;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_model():
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/6a11cb/brain.png", width=80)
    st.title("About")
    st.info(
        """
        **Tweet Depression Classifier**  
        Enter any tweet and instantly check if it expresses depressive sentiment.
        
        - Built with SVM & TF-IDF
        - For educational/demo use only
        """
    )
    st.markdown("---")
    st.write("Created with ❤️ using [Streamlit](https://streamlit.io/)")

# --- Main App ---
st.title("🧠 Tweet Depression Classifier")
st.markdown(
    "Detect depressive sentiment in tweets using AI. "
    "Paste a tweet below and click **Classify** to see the result."
)

tweet = st.text_area(
    "Enter tweet here:",
    placeholder="Type or paste a tweet...",
    height=120,
    max_chars=280
)

if st.button("Classify"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet to classify.")
    else:
        # Preprocess and vectorize
        tweet_vec = vectorizer.transform([tweet])
        prediction = model.predict(tweet_vec)[0]
        proba = model.predict_proba(tweet_vec)[0][int(prediction)] if hasattr(model, "predict_proba") else None

        if prediction == 1:
            st.error("😥 **Depressive Tweet Detected!**")
            if proba:
                st.write(f"Confidence: `{proba:.2%}`")
            st.markdown(
                "> If you or someone you know is struggling, please reach out to a mental health professional."
            )
        else:
            st.success("😊 **Not Depressive**")
            if proba:
                st.write(f"Confidence: `{proba:.2%}`")

# --- Footer ---
st.markdown(
    "<hr style='margin-top:2em;margin-bottom:1em'>"
    "<small>🚨 This tool is for educational purposes only and not a substitute for professional help.</small>",
    unsafe_allow_html=True
)
