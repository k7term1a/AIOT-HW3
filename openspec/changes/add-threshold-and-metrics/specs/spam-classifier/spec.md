## ADDED Requirements

### Requirement: Adjustable Decision Threshold
The system SHALL allow users to set a decision threshold in the range [0.1, 0.9] applied to the spam probability/score for classifying messages as spam vs ham.

#### Scenario: Threshold at 0.7 reduces false positives
- **WHEN** the user sets the threshold to 0.7
- **THEN** the app classifies messages as spam when `spam_score >= 0.7`
- **AND** the confusion matrix, precision, recall, and F1 update accordingly

#### Scenario: Threshold at 0.3 reduces false negatives
- **WHEN** the user sets the threshold to 0.3
- **THEN** the app classifies messages as spam when `spam_score >= 0.3`
- **AND** the metrics recalculate for the 0.3 threshold

### Requirement: Rich Model Evaluation
The system SHALL display ROC curve with AUC and Precision-Recall curve with AUC for the current model using the test set scores.

#### Scenario: Curves render from test set scores
- **WHEN** the model is trained
- **THEN** the app plots ROC and PR curves computed from test set scores

The system SHALL plot precision and recall as a function of threshold.

#### Scenario: Threshold sweeps
- **WHEN** the evaluation section is open
- **THEN** a plot shows precision and recall across thresholds 0.0–1.0

### Requirement: Downloadable Scored Test Set
The system SHALL provide a CSV of test examples with fields: text, true_label, spam_score, pred_at_threshold.

#### Scenario: Export results
- **WHEN** the user clicks “Download scored test set”
- **THEN** the CSV downloads with headers and UTF-8 encoding
