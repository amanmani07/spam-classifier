"""
predict.py
----------
Quick command-line tester for the trained model. Useful for sanity-checking
before you bother spinning up the Streamlit app.

Run:  python predict.py
Then type a message and press Enter. Type 'quit' to exit.
"""

import pickle
from preprocess import clean_text

with open("model/spam_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def predict(message: str):
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "SPAM" if pred == 1 else "HAM (not spam)"
    confidence = prob[pred] * 100
    return label, confidence


if __name__ == "__main__":
    print("Email Spam Classifier — CLI tester (type 'quit' to exit)\n")
    while True:
        msg = input("Enter a message: ").strip()
        if msg.lower() == "quit":
            break
        if not msg:
            continue
        label, confidence = predict(msg)
        print(f"  -> {label}  ({confidence:.1f}% confidence)\n")
