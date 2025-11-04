## Why
Current predictions use a fixed 0.5 threshold and provide limited evaluation. Stakeholders need to tune the decision threshold and review richer diagnostics (ROC/PR curves) to balance false positives/negatives for their context.

## What Changes
- Add adjustable decision threshold (0.1–0.9) with live recomputation of confusion matrix, precision, recall, and F1.
- Add ROC curve + AUC; Precision-Recall curve + AUC; precision/recall vs threshold plot.
- Add downloadable CSV of the scored test set (text, true_label, spam_score, pred_at_threshold).
- Non-breaking UI additions in `app.py`; refactor evaluation into helper utilities.

## Impact
- Affected specs: spam-classifier
- Affected code: `app.py` evaluation UI and utilities; expose decision scores from model.
