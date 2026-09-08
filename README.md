# 📧 Email Spam Classifier

A machine learning project that classifies email/SMS messages as **spam** or **not spam (ham)**,
built with scikit-learn (TF-IDF + Naive Bayes) and served through an interactive Streamlit web app.

**Test set results:** 97.6% accuracy · 99.2% precision · 82.6% recall · 90.1% F1 (spam class)

---

## 1. Project structure

```
spam-classifier/
├── data/
│   └── spam.csv          # labeled dataset (label, text)
├── model/                # created after training — saved model + vectorizer
├── preprocess.py         # shared text-cleaning logic
├── train_model.py        # trains the model, prints metrics, saves artifacts
├── predict.py            # command-line tester
├── app.py                # Streamlit web demo
├── requirements.txt
└── README.md
```

## 2. How it works (the short version)

1. **Data**: 5,572 real SMS messages labeled `ham` or `spam` (the classic "SMS Spam Collection" dataset).
2. **Cleaning**: lowercase → strip punctuation/numbers → remove stopwords → stem words (`preprocess.py`).
3. **Feature extraction**: `TfidfVectorizer` turns cleaned text into numeric vectors (word + bigram frequencies weighted by how distinctive they are).
4. **Model**: `MultinomialNB` (Naive Bayes) — the standard, fast, surprisingly strong baseline for text classification. It's also easy to explain in an interview, which matters for a portfolio piece.
5. **Evaluation**: accuracy, precision, recall, F1, and a confusion matrix on a held-out 20% test split.
6. **Serving**: the trained model + vectorizer are pickled and loaded by a Streamlit app for a live, clickable demo.

---

## 3. Setup in VS Code on Windows

### Step 1 — Get the project into a folder
Create a folder (e.g. `spam-classifier`) and copy all the files from this project into it, keeping the `data/spam.csv` path intact. Open that folder in VS Code (`File > Open Folder...`).

### Step 2 — Open a terminal in VS Code
`` Ctrl + ` `` (backtick) opens the integrated terminal. Make sure it's using **PowerShell** or **Command Prompt** (bottom-right of the terminal panel lets you pick).

### Step 3 — Create a virtual environment
```powershell
python -m venv venv
```

### Step 4 — Activate it
PowerShell:
```powershell
venv\Scripts\Activate.ps1
```
Command Prompt:
```cmd
venv\Scripts\activate.bat
```
> If PowerShell blocks the script with an execution-policy error, run this once:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

You should see `(venv)` appear at the start of your terminal prompt once it's active.

### Step 5 — Install dependencies
```powershell
pip install -r requirements.txt
```

### Step 6 — Select the interpreter in VS Code
`Ctrl+Shift+P` → "Python: Select Interpreter" → choose the one inside `.\venv\Scripts\python.exe`. This makes sure VS Code's linting/IntelliSense uses the same environment as your terminal.

---

## 4. Train the model
```powershell
python train_model.py
```
This prints accuracy/precision/recall/F1 and a confusion matrix, then saves
`model/spam_model.pkl` and `model/vectorizer.pkl`. Takes a few seconds.

## 5. Try it from the command line
```powershell
python predict.py
```
Type any message and it'll classify it live. Type `quit` to exit.

## 6. Run the web demo
```powershell
streamlit run app.py
```
This opens a browser tab at `http://localhost:8501` with a text box, example
messages, and a live classification with confidence score — this is the
version worth linking to from your portfolio/resume.

---

## 7. Understanding the numbers (good to know if asked about this project)

- **Precision is very high (99%)**: when the model says "spam," it's almost always right — few legitimate emails get wrongly flagged.
- **Recall is lower (83%)**: it misses some spam (calls it ham). This is a deliberate/typical trade-off for a spam filter — falsely blocking a real email is usually worse than letting one spam message through.
- The dataset is **imbalanced** (~13% spam), which is realistic and why accuracy alone is a bit misleading — precision/recall/F1 tell the real story. This is a good talking point if someone asks about your evaluation choices.

## 8. Ideas to extend it (nice additions for a portfolio project)

- Swap Naive Bayes for Logistic Regression or a linear SVM and compare metrics.
- Add a confusion-matrix / word-cloud visualization to `app.py`.
- Try `TfidfVectorizer(ngram_range=(1,3))` or character n-grams for obfuscated spam ("fr33 c@sh").
- Package it as a small Flask/FastAPI REST API instead of (or alongside) Streamlit.
- Add cross-validation in `train_model.py` instead of a single train/test split.

## 9. Deploying so you can link a live demo

1. Push this folder to a public GitHub repo (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and deploy the repo — point it at `app.py`.
3. You now have a live URL to put in your resume/portfolio next to the GitHub link.

### Pushing to GitHub
```powershell
git init
git add .
git commit -m "Email spam classifier: TF-IDF + Naive Bayes with Streamlit demo"
git branch -M main
git remote add origin https://github.com/<your-username>/spam-classifier.git
git push -u origin main
```

## 10. Dataset credit
"SMS Spam Collection" dataset — a well-known public dataset of labeled SMS messages, originally compiled for spam-filtering research and widely used for this exact type of project.
