import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load model and preprocessor

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")
    model = joblib.load("model.pkl")
    return preprocessor, model

preprocessor, model = load_artifacts()


# Streamlit page config

st.set_page_config(
    page_title="Bank Term Deposit Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Term Deposit Subscription Prediction")
st.write(
    "Enter customer details on the left and click **Predict** to see "
    "the probability that the customer will subscribe to a term deposit."
)


# Sidebar inputs

st.sidebar.header("Customer Information")

# 1. Demographics
st.sidebar.subheader("Demographics")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=40, step=1)

job = st.sidebar.selectbox(
    "Job",
    [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown"
    ],
)

marital = st.sidebar.selectbox(
    "Marital Status",
    ["married", "single", "divorced", "unknown"]
)

education = st.sidebar.selectbox(
    "Education",
    ["basic.4y", "basic.6y", "basic.9y", "high.school",
     "professional.course", "university.degree", "illiterate", "unknown"]
)

default = st.sidebar.selectbox(
    "Has credit in default?",
    ["no", "yes", "unknown"]
)

# 2. Financial
st.sidebar.subheader("Financial Status")
housing = st.sidebar.selectbox("Housing loan?", ["no", "yes", "unknown"])
loan = st.sidebar.selectbox("Personal loan?", ["no", "yes", "unknown"])

# 3. Contact & Campaign
st.sidebar.subheader("Contact & Campaign Details")
contact = st.sidebar.selectbox("Contact communication type", ["cellular", "telephone"])

month = st.sidebar.selectbox(
    "Last contact month",
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"]
)

day_of_week = st.sidebar.selectbox(
    "Last contact day of week",
    ["mon", "tue", "wed", "thu", "fri"]
)

campaign = st.sidebar.number_input(
    "Number of contacts performed during this campaign for this client",
    min_value=1, max_value=50, value=1, step=1
)

pdays = st.sidebar.number_input(
    "Days since last contact in previous campaign (999 = never contacted)",
    min_value=0, max_value=999, value=999, step=1
)

previous = st.sidebar.number_input(
    "Number of contacts before this campaign",
    min_value=0, max_value=50, value=0, step=1
)

poutcome = st.sidebar.selectbox(
    "Outcome of previous marketing campaign",
    ["nonexistent", "failure", "success", "unknown"]
)

# 4. Economic indicators
st.sidebar.subheader("Economic Indicators")
emp_var_rate = st.sidebar.number_input(
    "Employment variation rate (emp.var.rate)",
    value=1.1, format="%.1f"
)

cons_price_idx = st.sidebar.number_input(
    "Consumer price index (cons.price.idx)",
    value=93.994, format="%.3f"
)

cons_conf_idx = st.sidebar.number_input(
    "Consumer confidence index (cons.conf.idx)",
    value=-36.4, format="%.1f"
)

euribor3m = st.sidebar.number_input(
    "Euribor 3 month rate (euribor3m)",
    value=4.857, format="%.3f"
)

nr_employed = st.sidebar.number_input(
    "Number of employees (nr.employed)",
    value=5191.0, format="%.1f"
)


# Build input DataFrame

def build_input_df():
    """
    Build a single-row DataFrame that matches the training features.
    We removed 'duration' in training and added 'was_contacted_before'.
    """
    was_contacted_before = 0 if pdays == 999 else 1

    data = {
        "age": [age],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "default": [default],
        "housing": [housing],
        "loan": [loan],
        "contact": [contact],
        "month": [month],
        "day_of_week": [day_of_week],
        "campaign": [campaign],
        "pdays": [pdays],
        "previous": [previous],
        "poutcome": [poutcome],
        "emp.var.rate": [emp_var_rate],
        "cons.price.idx": [cons_price_idx],
        "cons.conf.idx": [cons_conf_idx],
        "euribor3m": [euribor3m],
        "nr.employed": [nr_employed],
        "was_contacted_before": [was_contacted_before],
    }

    return pd.DataFrame(data)


# Prediction section

st.subheader("Prediction")

if st.button("Predict Subscription Probability"):
    input_df = build_input_df()

    # Apply the same preprocessing as in training
    X_transformed = preprocessor.transform(input_df)

    # Get probability for class 1 (yes)
    prob_yes = model.predict_proba(X_transformed)[0, 1]

    # ----- CUSTOM THRESHOLD HERE -----
    threshold = 0.4
    pred_class = 1 if prob_yes >= threshold else 0
    # ---------------------------------

    label = "Will Subscribe" if pred_class == 1 else "Will Not Subscribe"

    st.markdown(f"### Result: **{label}**")
    st.write(f"**Probability of subscription (yes):** {prob_yes:.3f}")
    st.write(f"**Decision threshold used:** {threshold:.2f}")

    # A small interpretation
    if prob_yes >= 0.7:
        st.success("This customer is highly likely to subscribe. Consider prioritizing them in the campaign.")
    elif prob_yes >= threshold:
        st.info("This customer has a moderate chance to subscribe. A follow-up call or better offer may help.")
    else:
        st.warning("This customer is unlikely to subscribe. You may deprioritize them for this campaign.")

    st.write("---")
    st.write("### Input Summary")
    st.dataframe(input_df)
else:
    st.info("Fill out the details on the left and click **Predict Subscription Probability**.")
