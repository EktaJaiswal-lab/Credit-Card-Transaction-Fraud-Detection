from datetime import date, datetime, time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from fraud_features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, build_features


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "fraud_pipeline.joblib"

st.set_page_config(page_title="Fraud Detection Demo", page_icon="💳")
st.title("Credit Card Transaction Fraud Detection")
st.write(
    "Enter transaction and location details to see the model's prediction. "
    "This is an educational demo, not a financial decision tool."
)

if not MODEL_PATH.is_file():
    st.error("Model not found. Run `python train_model.py` before starting the app.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

with st.form("transaction"):
    st.subheader("Transaction")
    amount = st.number_input("Amount", min_value=0.0, value=50.0, step=1.0)
    transaction_date = st.date_input("Transaction date", value=date.today())
    transaction_time = st.time_input("Transaction time", value=time(12, 0))
    birth_date = st.date_input(
        "Cardholder date of birth", value=date(1985, 1, 1), max_value=date.today()
    )
    category = st.text_input("Transaction category", value="grocery_pos")
    merchant = st.text_input("Merchant", value="fraud_example merchant")

    st.subheader("Cardholder location")
    gender = st.selectbox("Gender", ["F", "M"])
    city = st.text_input("City", value="New York")
    state = st.text_input("State", value="NY", max_chars=2)
    postal_code = st.number_input("ZIP code", min_value=0, max_value=99999, value=10001)
    job = st.text_input("Job", value="Analyst")
    city_population = st.number_input(
        "City population", min_value=0, value=100000, step=1000
    )
    latitude = st.number_input("Cardholder latitude", value=40.7128, format="%.6f")
    longitude = st.number_input("Cardholder longitude", value=-74.0060, format="%.6f")

    st.subheader("Merchant location")
    merchant_latitude = st.number_input(
        "Merchant latitude", value=40.7306, format="%.6f"
    )
    merchant_longitude = st.number_input(
        "Merchant longitude", value=-73.9352, format="%.6f"
    )
    submitted = st.form_submit_button("Check transaction")

if submitted:
    transaction = pd.DataFrame(
        [
            {
                "amt": amount,
                "trans_date_trans_time": datetime.combine(
                    transaction_date, transaction_time
                ),
                "dob": datetime.combine(birth_date, time.min),
                "category": category.strip(),
                "merchant": merchant.strip(),
                "gender": gender,
                "city": city.strip(),
                "state": state.strip().upper(),
                "zip": postal_code,
                "job": job.strip(),
                "city_pop": city_population,
                "lat": latitude,
                "long": longitude,
                "merch_lat": merchant_latitude,
                "merch_long": merchant_longitude,
            }
        ]
    )
    features = build_features(transaction)[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    fraud_probability = float(model.predict_proba(features)[0, 1])
    prediction = int(model.predict(features)[0])

    if prediction:
        st.warning("Model prediction: potentially fraudulent")
    else:
        st.success("Model prediction: likely legitimate")
    st.metric("Estimated fraud probability", f"{fraud_probability:.1%}")
    st.caption(
        "Predictions are model estimates, not verified outcomes. "
        "The training pipeline uses class weighting, so probabilities may not "
        "be calibrated to real-world fraud rates."
    )
