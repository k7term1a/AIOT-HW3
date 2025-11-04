# Project Context

## Purpose
Build a Streamlit service for detecting spam/ham in short text (SMS/email) using datasets from Chapter03 of PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity. The app enables quick training, evaluation (accuracy/report/confusion matrix), and interactive single/batch predictions.

## Tech Stack
- Python 3.10+
- Streamlit 1.39
- scikit-learn 1.5 (TfidfVectorizer + MultinomialNB)
- pandas 2.2, numpy 2.1, requests 2.32
- Data inputs: GitHub raw URLs (auto candidates) or user-uploaded CSV/TSV

## Project Conventions

### Code Style
- PEP 8 with type hints on public functions
- Descriptive names; avoid one-letter identifiers
- Minimal inline comments; prefer clear, small functions
- Keep patches small and focused; avoid unrelated changes

### Architecture Patterns
- Single Streamlit entrypoint `app.py`
- Layered helpers inside `app.py`:
  - Data loading/normalization (`label`/`text`, `v1`/`v2`, 2-col TSV)
  - Model training/evaluation (cached)
  - Prediction utilities (single and batch)
- `st.cache_data` for datasets; `st.cache_resource` for trained model

### Testing Strategy
- Unit tests (future addition) with `pytest` for:
  - Dataset normalization edge cases
  - Predictive path sanity on small fixtures
  - App smoke test (import/launch without training)

### Git Workflow
- Branch naming: `feat/<slug>`, `fix/<slug>`, `chore/<slug>`, `docs/<slug>`
- Conventional Commits for messages
- PR review before merge; squash-merge to `main`

## Domain Context
- Binary classification of short messages (ham vs spam)
- Labels normalized to `ham`/`spam`
- Model optimized for speed and simplicity; suitable for CPU-only environments

## Important Constraints
- Network may be restricted; support offline via file upload
- Respect dataset licensing; do not persist PII beyond in-memory usage
- Keep training fast; avoid heavy models by default

## External Dependencies
- GitHub raw content for default datasets
- Python libs: Streamlit, scikit-learn, pandas, numpy, requests
