"""
preprocess.py
-------------
Shared text-cleaning logic used by both train_model.py and predict.py / app.py.
Keeping this in one file guarantees training and inference use the IDENTICAL
preprocessing steps — a very common bug source in ML projects is training on
text cleaned one way and predicting on text cleaned another way.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download once; nltk caches it locally after the first run.
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Lowercase, strip punctuation/numbers, remove stopwords, and stem.
    Example: "Free entry!! WIN cash NOW 2023" -> "free entri win cash"
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # keep letters/spaces only
    tokens = text.split()
    tokens = [_stemmer.stem(t) for t in tokens if t not in _stop_words and len(t) > 1]
    return " ".join(tokens)
