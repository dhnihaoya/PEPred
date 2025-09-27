# CLI Tool of PEPred

A command-line tool for the final logistic regression model for batch prediction and model inspection.

Model path: `final_model/final_lr_model.joblib`

## Dependencies

- Python 3.8+
- Required packages: `pandas`, `scikit-learn`, `joblib`

Install:
```bash
pip install pandas scikit-learn joblib
```

## Usage

### 1. View Model Information

```bash
python final_model_cli.py info
```

Optional parameters:
- `--model`: Model file path (default: `final_model/final_lr_model.joblib`)

### 2. Export Expected Features

```bash
python final_model_cli.py features --output features_list.csv
```

- Without `--output`, prints feature names to console
- Feature names must match input CSV column names exactly

### 3. Batch Prediction

```bash
python final_model_cli.py predict --input data/bestcombo_8feas.csv --output predictions.csv --threshold 0.5
```

Parameters:
- `--input` (required): Input CSV with all required feature columns
- `--output` (optional): Output CSV path (default: adds `_pred.csv` suffix to input)
- `--threshold` (optional): Positive class threshold (default: 0.5)
- `--id-column` (optional): ID column name to preserve in output

Input requirements:
- Column names must match all expected features (use `features` command to get list)
- UTF-8 encoding recommended

Output columns:
- `prob_positive`: Model probability output (0-1)
- `pred_label`: Binary prediction (0/1) based on threshold
- ID column (if specified and present in input)

## Example Commands

```bash
# View model info
python final_model_cli.py info

# Export feature list
python final_model_cli.py features --output features.csv

# Run predictions
python final_model_cli.py predict --input data/input.csv --output results.csv --threshold 0.5
```

## Troubleshooting

- Model file not found: Check `--model` path
- Missing feature columns: Use `features` command to get required columns
- Prediction errors: Verify scikit-learn/joblib versions are compatible