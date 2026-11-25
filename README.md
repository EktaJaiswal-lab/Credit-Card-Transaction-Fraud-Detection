# Credit Card Transaction Fraud Detection

An educational machine-learning project that explores transaction data, engineers
time, age, and geographic-distance features, and trains a scikit-learn classification
pipeline to distinguish potentially fraudulent transactions.

## Project files

- `ML_Project_1.ipynb` — exploratory analysis and model experiments.
- `train_model.py` — trains and evaluates the reusable preprocessing/model pipeline.
- `fraud_features.py` — shared feature engineering for training and app predictions.
- `app.py` — Streamlit interface for trying one transaction at a time.
- `fraudTrain.csv` — training data (keep local; do not commit large or sensitive data).

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
encoding as training. Train the model locally before launching the app.

## Deploy and share

1. Create a GitHub repository and push the source files, README, and requirements.
   Do not commit the CSV files, `.venv`, or private transaction data.
2. Train the model and make the generated `artifacts` available to the deployment.
   If the model is too large for normal Git hosting, use Git LFS or download it from
   an approved artifact store during deployment.
3. For Streamlit Community Cloud, create an app from the repository, select
   `app.py`, and deploy. Ensure the model artifact is available at
   `artifacts/fraud_pipeline.joblib`; Community Cloud cannot train from a local CSV
   that was not included in the repository.
4. Add the GitHub repository and live app URL here, then share those links in your
   portfolio or LinkedIn post.

**GitHub repository:** _Add link after publishing._

**Deployed app:** _Add link after deployment._

## Limitations

This is a portfolio demonstration, not a production fraud decision system. The model
uses class weighting, which can affect probability calibration. Do not enter real
cardholder personal or financial information. Evaluate performance, calibration,
privacy, and operational requirements before any real-world use.
