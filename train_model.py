import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, build_features


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
REQUIRED_COLUMNS = {
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
}
TEST_CHUNK_SIZE = 50_000
MODEL_INPUT_COLUMNS = [
    "trans_date_trans_time",
    "dob",
    "amt",
    "zip",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    *CATEGORICAL_FEATURES,
    "is_fraud",
]


def inspect_dataset(path):
    columns = list(pd.read_csv(path, nrows=0).columns)
    missing_columns = sorted(REQUIRED_COLUMNS - set(columns))
    if missing_columns:
        raise ValueError(
            f"Dataset QC failed for {path.name}; missing columns: {missing_columns}"
        )
    stats = {
        "columns": columns,
        "rows": 0,
        "missing": {},
        "fraud_counts": {},
        "invalid_targets": 0,
        "invalid_dates": 0,
        "invalid_birth_dates": 0,
        "birth_dates_after_transaction": 0,
        "invalid_amounts": 0,
        "invalid_coordinates": 0,
        "duplicate_transaction_ids": 0,
        "transaction_start": None,
        "transaction_end": None,
        "transaction_ids": set(),
    }

    for chunk in pd.read_csv(path, chunksize=100_000):
        stats["rows"] += len(chunk)
        for column, count in chunk.isna().sum().items():
            stats["missing"][column] = stats["missing"].get(column, 0) + int(count)
        for value, count in chunk["is_fraud"].value_counts(dropna=False).items():
            stats["fraud_counts"][value] = (
                stats["fraud_counts"].get(value, 0) + int(count)
            )
        stats["invalid_targets"] += int(
            (~chunk["is_fraud"].isin([0, 1])).sum()
        )

        transaction_dates = pd.to_datetime(
            chunk["trans_date_trans_time"], errors="coerce"
        )
        birth_dates = pd.to_datetime(chunk["dob"], errors="coerce")
        stats["invalid_dates"] += int(transaction_dates.isna().sum())
        stats["invalid_birth_dates"] += int(birth_dates.isna().sum())
        stats["birth_dates_after_transaction"] += int(
            (birth_dates > transaction_dates).sum()
        )
        first_date = transaction_dates.min()
        last_date = transaction_dates.max()
        if pd.notna(first_date) and (
            stats["transaction_start"] is None
            or first_date < stats["transaction_start"]
        ):
            stats["transaction_start"] = first_date
        if pd.notna(last_date) and (
            stats["transaction_end"] is None
            or last_date > stats["transaction_end"]
        ):
            stats["transaction_end"] = last_date

        amount = pd.to_numeric(chunk["amt"], errors="coerce")
        stats["invalid_amounts"] += int(
            (amount.isna() | (amount < 0)).sum()
        )
        coordinates = {
            "lat": (-90, 90),
            "long": (-180, 180),
            "merch_lat": (-90, 90),
            "merch_long": (-180, 180),
        }
        for column, (minimum, maximum) in coordinates.items():
            values = pd.to_numeric(chunk[column], errors="coerce")
            stats["invalid_coordinates"] += int(
                (values.isna() | ~values.between(minimum, maximum)).sum()
            )

        ids = chunk["trans_num"].dropna().astype(str)
        stats["duplicate_transaction_ids"] += int(ids.duplicated().sum())
        unique_ids = set(ids)
        stats["duplicate_transaction_ids"] += len(
            stats["transaction_ids"].intersection(unique_ids)
        )
        stats["transaction_ids"].update(unique_ids)

    return stats


def validate_datasets(train_stats, test_stats):
    issues = []
    for name, stats in (
        ("fraudTrain.csv", train_stats),
        ("fraudTest.csv", test_stats),
    ):
        missing_columns = sorted(REQUIRED_COLUMNS - set(stats["columns"]))
        if missing_columns:
            issues.append(f"{name} is missing required columns: {missing_columns}")
        extra_columns = sorted(
            set(stats["columns"])
            - REQUIRED_COLUMNS
            - {"first", "last", "Unnamed: 0"}
        )
        if extra_columns:
            issues.append(f"{name} has unexpected columns: {sorted(extra_columns)}")
        missing_values = {
            column: count for column, count in stats["missing"].items() if count
        }
        if missing_values:
            issues.append(f"{name} contains missing values: {missing_values}")
        if stats["invalid_targets"]:
            issues.append(f"{name} has invalid or missing target values.")
        if stats["invalid_dates"] or stats["invalid_birth_dates"]:
            issues.append(f"{name} has invalid transaction or birth dates.")
        if stats["birth_dates_after_transaction"]:
            issues.append(f"{name} has birth dates after transaction dates.")
        if stats["invalid_amounts"]:
            issues.append(f"{name} has invalid or negative transaction amounts.")
        if stats["invalid_coordinates"]:
            issues.append(f"{name} has invalid latitude or longitude values.")
        if stats["duplicate_transaction_ids"]:
            issues.append(f"{name} has duplicate transaction IDs.")
        if not stats["rows"]:
            issues.append(f"{name} is empty.")
        if set(stats["fraud_counts"]) != {0, 1}:
            issues.append(f"{name} must contain both fraud target classes (0 and 1).")

    if train_stats["columns"] != test_stats["columns"]:
        issues.append("Training and test CSV schemas or column order do not match.")
    if (
        train_stats["transaction_end"] is None
        or test_stats["transaction_start"] is None
        or train_stats["transaction_end"] >= test_stats["transaction_start"]
    ):
        issues.append("Training dates must end before test dates begin.")
    overlap = train_stats["transaction_ids"].intersection(
        test_stats["transaction_ids"]
    )
    if overlap:
        issues.append(
            f"Training and test sets share {len(overlap):,} transaction IDs."
        )

    if issues:
        raise ValueError("Dataset QC failed:\n- " + "\n- ".join(issues))

    for name, stats in (
        ("fraudTrain.csv", train_stats),
        ("fraudTest.csv", test_stats),
    ):
        fraud_count = stats["fraud_counts"][1]
        fraud_rate = fraud_count / stats["rows"]
        print(
            f"QC passed: {name}: {stats['rows']:,} rows, "
            f"{fraud_count:,} fraud ({fraud_rate:.3%}), "
            f"dates {stats['transaction_start']} to {stats['transaction_end']}."
        )
    print("QC passed: matching schemas, chronological split, and unique transaction IDs.")


def main():
    dataset_path = ROOT / "fraudTrain.csv"
    test_dataset_path = ROOT / "fraudTest.csv"
    if not dataset_path.is_file() or not test_dataset_path.is_file():
        raise FileNotFoundError(
            "Both fraudTrain.csv and fraudTest.csv must be present in the project folder."
        )

    train_stats = inspect_dataset(dataset_path)
    test_stats = inspect_dataset(test_dataset_path)
    validate_datasets(train_stats, test_stats)
    del train_stats, test_stats

    data = pd.read_csv(dataset_path, usecols=MODEL_INPUT_COLUMNS)
    if "is_fraud" not in data.columns:
        raise ValueError("fraudTrain.csv must contain the 'is_fraud' target column.")

    X = build_features(data)
    y = data["is_fraud"]
    missing = sorted(set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES) - set(X.columns))
    if missing:
        raise ValueError(f"Training data is missing required features: {missing}")

    X = X[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    tol=1e-3,
                    solver="saga",
                    random_state=42,
                ),
            ),
        ]
    )
    print(f"Training on {len(X):,} rows from fraudTrain.csv.")
    model.fit(X, y)
    del X, data

    y_test = []
    predictions = []
    probabilities = []
    test_rows = 0
    for chunk in pd.read_csv(
        test_dataset_path, usecols=MODEL_INPUT_COLUMNS, chunksize=TEST_CHUNK_SIZE
    ):
        test_rows += len(chunk)
        chunk_y = chunk["is_fraud"]
        chunk_X = build_features(chunk)[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
        y_test.extend(chunk_y.tolist())
        predictions.extend(model.predict(chunk_X).tolist())
        probabilities.extend(model.predict_proba(chunk_X)[:, 1].tolist())

    print(f"Evaluating on {test_rows:,} rows from fraudTest.csv.")
    print(classification_report(y_test, predictions, zero_division=0))
    print(
        "Average Precision (AP): "
        f"{average_precision_score(y_test, probabilities):.4f}"
    )

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "fraud_pipeline.joblib")
    with (ARTIFACTS / "feature_columns.json").open("w", encoding="utf-8") as file:
        json.dump(CATEGORICAL_FEATURES + NUMERICAL_FEATURES, file, indent=2)
    print(f"Saved model and feature list to {ARTIFACTS}")


if __name__ == "__main__":
    main()
