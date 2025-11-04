## ADDED Requirements

### Requirement: Confusion Matrix Heatmap
The system SHALL render the confusion matrix as a heatmap with color intensity proportional to counts and numeric annotations within each cell.

#### Scenario: Heatmap after training
- **WHEN** the model finishes training on the dataset
- **THEN** the app displays a 2x2 heatmap labeled with `Actual` (rows) and `Predicted` (columns)
- **AND** each cell shows the count as text

### Requirement: Label Distribution Chart
The system SHALL visualize the label distribution (ham vs spam) as a bar chart.

#### Scenario: Bar chart of labels
- **WHEN** the dataset is loaded
- **THEN** a bar chart shows counts for `ham` and `spam`

### Requirement: Quick-Fill Examples
The system SHALL provide buttons to auto-fill example texts for spam and ham in the prediction input area.

#### Scenario: Fill spam example
- **WHEN** the user clicks the "填入 Spam 範例" button
- **THEN** the input text area populates with a spam-like example

#### Scenario: Fill ham example
- **WHEN** the user clicks the "填入 Ham 範例" button
- **THEN** the input text area populates with a ham-like example

## MODIFIED Requirements

### Requirement: Simplified Input and Data Source
The system SHALL remove dataset source selection and file upload controls and SHALL default to loading the local Chapter03 dataset path.

#### Scenario: No upload/source UI
- **WHEN** the app loads
- **THEN** there is no dataset source selector or upload control
- **AND** the app attempts to load `Hands-On-Artificial-Intelligence-for-Cybersecurity/Chapter03/datasets/sms_spam_no_header.csv`
- **AND** if the file is missing, a clear error message is displayed

### Requirement: Robust Dataset Normalization
The system SHALL handle header-less CSVs using numeric column indices and drop any `Unnamed` columns without using `.str` accessors on the column index.

#### Scenario: Header-less CSV normalizes to label/text
- **WHEN** loading a two-column CSV with `header=None`
- **THEN** the dataset normalizes to columns `label` and `text` without raising a `.str` error

## REMOVED Requirements

### Requirement: File Upload for Dataset
**Reason**: Simplified UX to a single, default local dataset flow
**Migration**: Users copy or prepare the dataset locally at the default path
