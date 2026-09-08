"""
train_model.py
---------------
End-to-end training pipeline for the Email/SMS Spam Classifier.

Run:  python train_model.py

What it does:
  1. Loads data/spam.csv
  2. Cleans the text (preprocess.py)
  3. Converts text -> TF-IDF numeric features
  4. Trains a Multinomial Naive Bayes classifier
  5. Evaluates on a held-out test set
  6. Saves the trained model + vectorizer to model/ so app.py / predict.py can reuse them
"""

import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from preprocess import clean_text

DATA_PATH = "data/spam.csv"
MODEL_DIR = "model"


def main():
    # 1. Load data ---------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["label", "text"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    print(f"Loaded {len(df)} rows | spam={df['label_num'].sum()} ham={(df['label_num']==0).sum()}")

    # 2. Clean text ----------------------------------------------------------
    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    # 3. Train/test split (stratified so spam ratio is preserved in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label_num"],
        test_size=0.2,
        random_state=42,
        stratify=df["label_num"],
    )

    # 4. Vectorize (TF-IDF) --------------------------------------------------
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Train ----------------------------------------------------------------
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # 6. Evaluate ---------------------------------------------------------------
    y_pred = model.predict(X_test_vec)
    print("\n=== Evaluation on held-out test set ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 score : {f1_score(y_test, y_pred):.4f}")
    print("\nConfusion matrix (rows=actual, cols=predicted) [ham, spam]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nFull report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # 7. Save artifacts -----------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "spam_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"\nSaved model + vectorizer to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
