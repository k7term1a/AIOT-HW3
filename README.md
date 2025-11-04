# Spam Detector (Streamlit)

Live app: https://k7-aiot-hw3.streamlit.app/

This Streamlit app builds a simple spam/ham text classifier using a local dataset from Chapter03 of the repository:

PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity

The app automatically loads a dataset with this priority:

- Local file: `datasets/sms_spam_no_header.csv`
- URL: Packt GitHub raw file
- Built-in tiny sample (demo only)

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

4) The app auto-loads the dataset (local → URL → built-in). The UI shows dataset summary, evaluation metrics (accuracy, classification report, confusion matrix), ROC/PR curves, and a text area to classify a single message.

## Notes

- The model is a simple `TfidfVectorizer` + `MultinomialNB` pipeline.
- Training happens in-memory on app start and results are cached.
- No manual selection needed; it falls back automatically.
- The built-in sample is small and intended only for demonstration, not for benchmarking.
