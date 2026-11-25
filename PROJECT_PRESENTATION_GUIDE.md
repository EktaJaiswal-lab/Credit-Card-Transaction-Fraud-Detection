# Project Presentation and Video Guide

This guide is a recording plan and speaking script for the Credit Card Transaction
Fraud Detection project. It follows the code and the reported evaluation results.
Adjust the wording to sound natural, and replace any personal/course details before
recording.

## Project links

- **GitHub repository:** [Credit Card Transaction Fraud Detection](https://github.com/EktaJaiswal-lab/Credit-Card-Transaction-Fraud-Detection)
- **Live Streamlit app:** [Open the Fraud Detection app](https://credit-card-transaction-fraud-detection-pae7igwgfphrj4ukc9gph8.streamlit.app/)
- **Notebook:** `ML_Project_1.ipynb`
- **Training and evaluation:** `train_model.py`
- **Shared feature engineering:** `fraud_features.py`
- **Streamlit interface:** `app.py`

## Suggested video structure

Target **8–10 minutes**. Show your screen while explaining the project; avoid reading
every line of code. The timings are flexible.

| Time | Section |
| --- | --- |
| 0:00–0:40 | Introduction and objective |
| 0:40–1:40 | Dataset and fraud-class imbalance |
| 1:40–2:30 | Data quality checks |
| 2:30–3:20 | Notebook exploration |
| 3:20–4:20 | Feature engineering |
| 4:20–5:20 | Preprocessing and model |
| 5:20–6:30 | Test-set results and interpretation |
| 6:30–7:50 | Live app demonstration |
| 7:50–8:40 | Deployment, limitations, and conclusion |

## Before recording

1. Open the GitHub repository and deployed app in separate browser tabs.
2. Confirm the app loads and the model prediction form is visible.
3. If you want to show notebook charts, open `ML_Project_1.ipynb`, select the
   **Python (Fraud Detection)** kernel, and run the relevant cells. Notebook outputs
   were cleared before publishing, so charts will appear after the cells run.
4. Have the reported `fraudTest.csv` metrics below ready to show. Do not claim these
   are the notebook's random-split results.
5. Do not display or enter real cardholder or personal data. The CSV datasets remain
   local and are intentionally not in the public repository.

## Detailed narration and screen actions

### 1. Introduction and objective (0:00–0:40)

**Show:** Project title, then the GitHub repository.

**Suggested script:**

> Hello, I’m [your name]. This is my Credit Card Transaction Fraud Detection
> project. The goal is to use transaction and location information to estimate
> whether a transaction may be fraudulent. I built an end-to-end workflow: data
> quality checks, feature engineering, model training and evaluation, a saved
> scikit-learn pipeline, and a Streamlit app. This is a learning and portfolio
> demonstration, not a production banking system.

**Transition:** Explain what data the model learns from.

### 2. Dataset and class imbalance (0:40–1:40)

**Show:** The project folder or the notebook's initial data-loading/EDA cells.

**Suggested script:**

> The project uses two separate files. `fraudTrain.csv` contains 1,296,675
> transactions, including 7,506 fraud cases. `fraudTest.csv` contains 555,719 later
> transactions, including 2,145 fraud cases. The target column is `is_fraud`, where
> 1 means fraud and 0 means non-fraud.
>
> Fraud is rare: it represents about 0.579% of the training data and 0.386% of the
> test data. That imbalance matters. A model can have high accuracy simply by
> predicting “not fraud,” so I focus on fraud precision and recall as well as the
> overall metrics.

**Important context:**

- `fraudTrain.csv` covers 2019-01-01 through 2020-06-21 12:13:37.
- `fraudTest.csv` follows chronologically, from 2020-06-21 12:14:25 through
  2020-12-31.
- The test fraud rate is lower than the training rate. This time-related difference
  is retained as part of the evaluation rather than balancing or resampling the
  test set.
- The data files are excluded from GitHub. They must be present locally to retrain
  the model, but the deployed app only needs the saved model artifact.

### 3. Data quality checks (1:40–2:30)

**Show:** `train_model.py`, especially `inspect_dataset` and `validate_datasets`.

**Suggested script:**

> Before fitting the model, the training script performs quality checks on both
> files. It checks the required schema, missing values, target values, date parsing,
> birth dates that occur after the transaction, negative or invalid amounts,
> invalid coordinates, and duplicate transaction IDs. It also checks that the two
> files have matching schemas, share no transaction IDs, and that the test dates
> come after the training dates.
>
> These checks pass on the supplied data. The scan is performed in chunks so the
> entire CSV does not need to be held in memory just for quality checks.

**Screen detail:** The QC scanner reads 100,000 rows at a time. Test predictions
later run in 50,000-row chunks.

### 4. Notebook exploration (2:30–3:20)

**Show:** `ML_Project_1.ipynb` section headings, then one or two EDA charts.

**Suggested script:**

> In the notebook I explore the data, check its columns and types, examine fraud
> counts, compare transaction amounts and categories, and review numeric
> correlations. The notebook also contains model experiments and visualizations.
> This exploratory work helps me understand the features and the imbalance before
> creating the reusable training script.

**Important distinction to explain:**

> The notebook includes an exploratory random train/test split of the training
> file. The final reusable `train_model.py` workflow does not use that split for its
> reported test results. It fits on the full `fraudTrain.csv` file and evaluates on
> the separate, later-in-time `fraudTest.csv` file.

Do not compare a notebook experiment score with the final test report as if they
came from the same evaluation split.

### 5. Feature engineering (3:20–4:20)

**Show:** `fraud_features.py`, function `build_features`.

**Suggested script:**

> I convert the transaction timestamp into hour, day, month, and weekday features.
> I calculate the customer's age at the transaction date from the transaction date
> and date of birth. I also calculate the distance between the customer's location
> and the merchant's location using the Haversine formula, measured in kilometers.
>
> The model keeps transaction amount, ZIP code, city population, customer and
> merchant coordinates, the engineered time/age/distance values, and categorical
> details such as merchant, category, gender, city, state, and job. I remove fields
> such as card number, first and last name, street, transaction number, raw date of
> birth, Unix timestamp, and the CSV index. These identifiers and direct personal
> details are not model features.

**Model input columns:**

- **Categorical (6):** `merchant`, `category`, `gender`, `city`, `state`, `job`.
- **Numerical (13):** `amt`, `zip`, `lat`, `long`, `city_pop`, `merch_lat`,
  `merch_long`, `transaction_hour`, `transaction_day`, `transaction_month`,
  `transaction_weekday`, `customer_age`, `distance_km`.

The same feature function is called by the training script and the Streamlit app to
keep training-time and prediction-time feature construction consistent.

### 6. Preprocessing and model (4:20–5:20)

**Show:** Pipeline setup in `train_model.py`.

**Suggested script:**

> Preprocessing is inside a scikit-learn pipeline, so it is fitted with the training
> data and reused unchanged for the test data and app inputs. Numeric columns use
> median imputation and standard scaling. Categorical columns use most-frequent
> imputation and one-hot encoding. The encoder ignores categories it did not see
> during training and groups infrequent categories using a minimum frequency of
> 20.
>
> The final classifier is logistic regression with balanced class weights. The
> class weighting is used to pay more attention to the rare fraud class. The fitted
> preprocessing and classifier are saved together as
> `artifacts/fraud_pipeline.joblib`; the ordered raw feature list is also written to
> `artifacts/feature_columns.json`.

**Exact training settings:**

- `class_weight="balanced"`
- `max_iter=2000`
- `tol=1e-3`
- `solver="saga"`
- `random_state=42`

### 7. Test-set results and interpretation (5:20–6:30)

**Show:** The report from running `python train_model.py` or this table.

| Class/measure | Precision | Recall | F1 score | Support |
| --- | ---: | ---: | ---: | ---: |
| Non-fraud (0) | 1.00 | 0.92 | 0.96 | 553,574 |
| Fraud (1) | 0.03 | 0.55 | 0.05 | 2,145 |

**Overall accuracy:** 0.92 across 555,719 transactions.

**Suggested script:**

> On the separate test file, the fraud-class recall is 0.55. That means the model
> catches about 55% of the fraud cases in this test set, while missing the
> remainder. Fraud precision is 0.03, meaning only about 3% of the transactions
> flagged by this model are actually fraud. That is a substantial false-alert
> limitation.
>
> Accuracy is 0.92, but an always-non-fraud baseline would have about 99.61%
> accuracy because the data is so imbalanced. That baseline would catch no fraud,
> which is why accuracy by itself does not tell the whole story. The next modeling
> step would be to select an operating threshold using a validation set from the
> training period, and then leave this later test set untouched for final
> evaluation.

The classifier's reported class predictions use its default decision threshold.
Class weighting can affect probability calibration, so the app's displayed
probability should be described as a model estimate, not a calibrated real-world
fraud rate. `train_model.py` also prints Average Precision; include that number in
the video only if you have the output from your actual run.

### 8. Live Streamlit app demonstration (6:30–7:50)

**Show:** Open the deployed app:

<https://credit-card-transaction-fraud-detection-pae7igwgfphrj4ukc9gph8.streamlit.app/>

**Suggested narration and actions:**

1. Introduce the form:

   > This is the deployed Streamlit interface. A user can enter transaction details
   > and relevant customer and merchant location information.

2. Point out the inputs: amount, transaction date and time, date of birth,
   category, merchant, gender, city, state, ZIP code, job, city population, and
   customer and merchant coordinates.
3. Use clearly synthetic demo values, then select **Check transaction**.
4. Explain the output:

   > The app creates the engineered features, loads the saved pipeline, and shows
   > the model's predicted class and estimated fraud probability. The labels say
   > “potentially fraudulent” or “likely legitimate” because this is a prediction,
   > not a verified outcome.

5. Point out the on-screen notice that probabilities may not be calibrated.

Do not enter real card numbers, names, addresses, or other personal or financial
information. The app is for demonstration and does not replace a bank's fraud
monitoring or review process.

### 9. GitHub and deployment (7:50–8:20)

**Show:** GitHub repository and the app link.

**Suggested script:**

> The source code, notebook, requirements, README, and saved model are in GitHub.
> The app is deployed on Streamlit Community Cloud from the repository. The large
> CSV files are intentionally not uploaded; they are needed only to retrain the
> model, not to serve predictions from the committed pipeline.

The saved model was created with pinned package versions in `requirements.txt`,
including scikit-learn 1.9.1, to keep the deployment compatible with the serialized
pipeline.

### 10. Limitations and closing (8:20–9:00)

**Suggested script:**

> This project demonstrates an end-to-end machine-learning workflow, but it is not
> production ready. The low fraud precision means many alerts would be false
> positives. The training and test periods also have different fraud rates, and
> class weighting can affect probability calibration. A real system would need
> threshold selection based on investigation capacity and fraud costs, more
> validation over time, calibration checks, privacy controls, and monitoring for
> data drift.
>
> This project helped me practice data quality checks, feature engineering,
> scikit-learn pipelines, imbalanced classification, model persistence, Streamlit,
> and cloud deployment. Thank you for watching.

## Reproducing the project locally

Run these commands from the project directory in Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

`fraudTrain.csv` and `fraudTest.csv` must be present in the project directory to
retrain and evaluate. The app can run without the CSVs when
`artifacts/fraud_pipeline.joblib` is present.

## Common questions for the video

**Why not use accuracy alone?**

Fraud cases are less than 1% of both datasets. A classifier can appear highly
accurate while missing every fraud case.

**Why use separate training and test files?**

The supplied test file covers a later time period. This is closer to asking how the
model performs on transactions that happen after the training period than randomly
splitting the same period.

**Why is fraud precision low?**

Fraud is rare and the model uses balanced class weights to improve attention to the
minority class. At the current default decision rule, the trade-off favors catching
some fraud at the cost of many false alerts. The threshold should be chosen using
business costs and a separate validation set.

**Does the app prove a transaction is fraud?**

No. It reports a machine-learning estimate, not a confirmed investigation result.

**Why is the dataset not in GitHub?**

The files are large and contain sensitive-looking transaction and personal fields.
They are excluded to avoid publishing the raw records. The committed model artifact
is sufficient for the deployed app to make demonstration predictions.
