# Credit Card Transaction Fraud Detection

An educational machine-learning project that explores transaction data, engineers
time, age, and geographic-distance features, and trains a scikit-learn classification
pipeline to distinguish potentially fraudulent transactions.

## Project files

- `ML_Project_1.ipynb` — exploratory analysis and model experiments.
- `train_model.py` — trains and evaluates the reusable preprocessing/model pipeline.
- `fraud_features.py` — shared feature engineering for training and app predictions.
- `app.py` — Streamlit interface for trying one transaction at a time.
- `PROJECT_PRESENTATION_GUIDE.md` — detailed video narration, demo plan, metrics,
  and common questions.
- `fraudTrain.csv` and `fraudTest.csv` — training and test data (keep local; do not
  commit large or sensitive transaction data).

## Run locally (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

In VS Code, open the notebook and select the `.venv` Python interpreter as its kernel.
The notebook reads `fraudTrain.csv` from the project directory.

Before training, the script checks both CSVs for the expected schema, missing or
invalid values, duplicate transaction IDs, and chronological ordering. It fits on
`fraudTrain.csv` and evaluates on `fraudTest.csv` in chunks to limit memory use.
It saves:

- `artifacts/fraud_pipeline.joblib` — preprocessing and fitted classifier.
- `artifacts/feature_columns.json` — the model's ordered input feature names.

The test set is later in time than the training set and has a different fraud rate;
this reflects temporal distribution shift and is reported rather than rebalanced.
The app loads the saved pipeline, so it uses the same imputing, scaling, and category
encoding as training. The trained pipeline is included in `artifacts/`; rerun
`python train_model.py` to retrain it.

## Test-set results

The current logistic-regression pipeline was evaluated on all 555,719 rows of
`fraudTest.csv` (2,145 fraud cases; 0.386% prevalence). At the classifier's default
0.5 decision threshold, the fraud-class metrics were:

| Metric | Fraud class |
| --- | ---: |
| Precision | 0.03 |
| Recall | 0.55 |
| F1 score | 0.05 |

Overall accuracy was 0.92, but predicting every transaction as legitimate would
reach about 99.61% accuracy because fraud is rare. Accuracy alone is therefore not a
useful measure of this model. At this threshold, only about 3% of flagged transactions
are fraud, so false alerts are a substantial limitation. Threshold selection should
be evaluated on a validation split from the training period; keep `fraudTest.csv` as
the final, untouched evaluation set.

## Deploy and share

1. The source, requirements, README, and trained pipeline are in the GitHub
   repository below. The CSV files are intentionally excluded; keep them local and
   do not publish private transaction data.
2. For Streamlit Community Cloud, create an app from the repository, select
   `app.py`, and deploy. The committed model artifact is at
   `artifacts/fraud_pipeline.joblib`; the cloud app does not need the training CSVs
   to serve predictions.
3. Add the live app URL here after deployment, then share it alongside the
   repository in your portfolio or LinkedIn post.

**GitHub repository:** [Credit Card Transaction Fraud Detection](https://github.com/EktaJaiswal-lab/Credit-Card-Transaction-Fraud-Detection)

**Deployed app:** [Open the Fraud Detection app](https://credit-card-transaction-fraud-detection-pae7igwgfphrj4ukc9gph8.streamlit.app/)

## Limitations

This is a portfolio demonstration, not a production fraud decision system. The model
uses class weighting, which can affect probability calibration. Do not enter real
cardholder personal or financial information. Evaluate performance, calibration,
privacy, and operational requirements before any real-world use.
