## Why
Improve usability and clarity by visualizing evaluation results (confusion matrix heatmap, label distribution chart), simplifying inputs (single text area, no upload/source selection), and adding quick-fill examples. Also fix a normalization bug for header-less CSVs where non-string column indices caused `.str` errors.

## What Changes
- Add confusion matrix heatmap visualization with value annotations.
- Add label distribution bar chart (ham vs spam).
- Add quick-fill buttons for spam/ham examples in the prediction area.
- Remove dataset source selection and file upload UI; always load the local Chapter03 dataset.
- Make dataset normalization robust to numeric column indices and drop `Unnamed` columns without using `.str` accessors.
- Add `matplotlib` dependency for plotting.

## Impact
- Affected specs: spam-classifier
- Affected code: `app.py` (UI, visualization, normalization), `requirements.txt` (matplotlib)
