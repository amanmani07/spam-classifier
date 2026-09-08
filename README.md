# 📧 Email Spam Classifier

An end-to-end **Machine Learning project** that classifies SMS and email messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing and supervised learning.

The project uses **TF-IDF feature extraction** with a **Multinomial Naive Bayes classifier** and provides an interactive **Streamlit web application** for real-time predictions.

### 🚀 Live Demo

🌐 **[Try the Email Spam Classifier Live](https://spam-classifier-gz6rrkt2wsnujvvcqepjtd.streamlit.app/)**

💻 **[View Source Code](https://github.com/amanmani07/spam-classifier)**

---

## 📊 Model Performance

The model was evaluated on a held-out test set and achieved:

| Metric    |     Score |
| --------- | --------: |
| Accuracy  | **97.6%** |
| Precision | **99.2%** |
| Recall    | **82.6%** |
| F1-Score  | **90.1%** |

The dataset contains approximately **5,572 labeled SMS messages**, with `ham` and `spam` classes.

> Precision, recall, and F1-score are reported alongside accuracy because the dataset is imbalanced and accuracy alone does not fully represent spam-classification performance.

---

## 🧠 How It Works

The classification pipeline consists of several stages:

```text
User Message
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Multinomial Naive Bayes
     ↓
Spam / Ham Prediction
     ↓
Confidence Score
```

### 1. Text Preprocessing

The input text is cleaned before being passed to the model:

* Convert text to lowercase
* Remove punctuation and unnecessary characters
* Remove stopwords
* Stem words
* Normalize the text

### 2. Feature Extraction

`TfidfVectorizer` converts the processed text into numerical feature vectors.

The implementation uses word-level TF-IDF features and n-grams to capture useful patterns within messages.

### 3. Classification

A **Multinomial Naive Bayes** classifier is used to predict whether the message is:

* 🚨 **Spam**
* ✅ **Ham / Not Spam**

### 4. Live Prediction

The trained model and TF-IDF vectorizer are saved as serialized artifacts and loaded by the Streamlit application.

Users can enter any message and receive a prediction instantly.

---

## 🖥️ Live Application

The Streamlit interface allows users to:

* Enter custom messages
* Analyze messages in real time
* View Spam/Ham predictions
* View prediction confidence
* Test the model using example messages

### Try it yourself

🌐 **Live Demo:**
https://spam-classifier-gz6rrkt2wsnujvvcqepjtd.streamlit.app/

Example:

```text
Input:
"Congratulations! You have won a free prize. Click here to claim."

Prediction:
🚨 SPAM

Confidence:
98%+
```

---

## 🛠️ Tech Stack

**Programming Language**

* Python

**Machine Learning**

* Scikit-learn
* Multinomial Naive Bayes
* TF-IDF

**Natural Language Processing**

* NLTK
* Text preprocessing
* Stopword removal
* Stemming

**Data Processing**

* Pandas
* NumPy

**Web Application**

* Streamlit

**Development**

* VS Code
* Git
* GitHub

---

## 📁 Project Structure

```text
spam-classifier/
│
├── data/
│   └── spam.csv
│
├── model/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
│
├── app.py
├── predict.py
├── preprocess.py
├── train_model.py
├── requirements.txt
└── README.md
```

### File Description

| File               | Purpose                           |
| ------------------ | --------------------------------- |
| `app.py`           | Streamlit web application         |
| `train_model.py`   | Model training and evaluation     |
| `predict.py`       | Command-line prediction interface |
| `preprocess.py`    | Text preprocessing functions      |
| `spam_model.pkl`   | Trained Naive Bayes model         |
| `vectorizer.pkl`   | Trained TF-IDF vectorizer         |
| `spam.csv`         | Labeled SMS dataset               |
| `requirements.txt` | Python dependencies               |

---

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/amanmani07/spam-classifier.git
cd spam-classifier
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Train the model

```powershell
python train_model.py
```

This will train the classifier and generate the model artifacts inside the `model/` directory.

### 5. Test from the terminal

```powershell
python predict.py
```

### 6. Launch the web application

```powershell
streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

---

## 📈 Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The high precision indicates that the model is highly reliable when identifying messages as spam, while the lower recall indicates that some spam messages may still be classified as legitimate.

This demonstrates an important machine-learning trade-off in spam detection: **reducing false positives versus catching more spam.**

---

## 🔬 Future Improvements

Potential improvements include:

* Compare Naive Bayes with Logistic Regression
* Add Linear SVM
* Experiment with character-level n-grams
* Detect obfuscated spam such as `fr33 c@sh`
* Add confusion-matrix visualization to the web app
* Add model comparison charts
* Build a REST API using FastAPI
* Add automated testing
* Containerize the application with Docker
* Add CI/CD using GitHub Actions
* Experiment with transformer-based NLP models

---

## 📚 Dataset

This project uses the **SMS Spam Collection** dataset, a publicly available dataset containing labeled SMS messages for spam classification research.

Dataset classes:

```text
ham  → legitimate message
spam → unwanted/spam message
```

---

## 👨‍💻 Author

**Amman Mani**

Bachelor of Science in Artificial Intelligence

🔗 **GitHub:**
https://github.com/amanmani07

🌐 **Live Project:**
https://spam-classifier-gz6rrkt2wsnujvvcqepjtd.streamlit.app/

---

⭐ If you found this project useful, consider giving the repository a star.
