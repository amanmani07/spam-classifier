"""
app.py
------
Streamlit web UI for the Email Spam Classifier. This is the piece you'll
link to from your portfolio / resume — it turns the model into something
a recruiter can click and try in their browser instead of reading code.

Run:  streamlit run app.py
"""

import pickle
import streamlit as st
from preprocess import clean_text

st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="centered")


@st.cache_resource
def load_artifacts():
    with open("model/spam_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


model, vectorizer = load_artifacts()

st.title("📧 Email / SMS Spam Classifier")
st.write(
    "A Naive Bayes classifier trained with TF-IDF features on 5,500+ labeled "
    "messages. Paste an email or text message below to see how it's classified."
)

examples = {
    "-- Select an example --": "",
    "Spam example": "CONGRATULATIONS! You've won a $1000 Walmart gift card. "
    "Click the link now to claim your prize before it expires!",
    "Ham example": "Hey, just checking if we're still meeting for coffee at 10am tomorrow?",
}
choice = st.selectbox("Try an example, or write your own below:", list(examples.keys()))

user_input = st.text_area(
    "Message to classify",
    value=examples[choice] if choice != "-- Select an example --" else "",
    height=150,
    placeholder="Paste an email or SMS message here...",
)

if st.button("Classify", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        if pred == 1:
            st.error(f"🚫 **Spam** — {prob[1] * 100:.1f}% confidence")
        else:
            st.success(f"✅ **Not Spam (Ham)** — {prob[0] * 100:.1f}% confidence")

        with st.expander("See how the text was processed"):
            st.write("**Cleaned text fed to the model:**")
            st.code(cleaned or "(empty after cleaning)")

st.divider()
st.caption(
    "Model: Multinomial Naive Bayes + TF-IDF | Trained on the SMS Spam Collection dataset | "
    "~97.6% test accuracy"
)
