# Spam Detector (Streamlit)

This Streamlit app builds a simple spam/ham text classifier using a local dataset from Chapter03 of the repository:

PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity

By default it loads `Hands-On-Artificial-Intelligence-for-Cybersecurity/Chapter03/datasets/sms_spam_no_header.csv` from your local clone.

## Quickstart

1) Create and activate a virtual environment (optional):

- Windows (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```
pip install -r requirements.txt
```

3) Run the app:

```
streamlit run app.py
```

4) The app automatically loads the local Chapter03 dataset and trains the model. The UI shows dataset summary, evaluation metrics (accuracy, classification report, confusion matrix), ROC/PR curves, and a text area to classify a single message.

## Notes

- The model is a simple `TfidfVectorizer` + `MultinomialNB` pipeline.
- Training happens in-memory on app start and results are cached.
- The UI is simplified: no file upload; use the single text area to test messages.
- Ensure the dataset exists at `Hands-On-Artificial-Intelligence-for-Cybersecurity/Chapter03/datasets/sms_spam_no_header.csv` relative to the project root.
