# Telco Churn (Minimal Deliverable)

This implementation is intentionally minimal and focused on your assignment requirements.

## What it produces

Only these **3 plots**:
1. `plots/01_churn_fraction_by_contract.png`
2. `plots/02_churn_fraction_by_seniorcitizen.png`
3. `plots/03_confusion_matrix.png`

Also produces:
- `metrics.txt` (confusion matrix, precision, recall, CV summary)
- `writeup.md` (short assignment writeup)

## Dataset placement

Put the Kaggle CSV at:

`decision-trees/churn/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Run

```bash
python3 churn_analysis.py
```

## Dependencies

```bash
pip3 install pandas scikit-learn matplotlib seaborn
```
