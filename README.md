# Spam Detector (Streamlit)

This Streamlit app builds a simple spam/ham text classifier using a local dataset from Chapter03 of the repository:

PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity

By default you can choose a dataset source at runtime:

- Use local file at `Hands-On-Artificial-Intelligence-for-Cybersecurity/Chapter03/datasets/sms_spam_no_header.csv`
- Upload a CSV/TSV file (two columns: label, text)
- Load from URL (default points to the Packt GitHub raw file)
- Use a tiny built-in sample dataset (for quick demo)

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

4) In the app, select your preferred data source. If the local file does not exist, use upload/URL/built-in options. The UI shows dataset summary, evaluation metrics (accuracy, classification report, confusion matrix), ROC/PR curves, and a text area to classify a single message.

## Notes

- The model is a simple `TfidfVectorizer` + `MultinomialNB` pipeline.
- Training happens in-memory on app start and results are cached.
- You can upload or use a URL if the local file is unavailable.
- The built-in sample is small and intended only for demonstration, not for benchmarking.
