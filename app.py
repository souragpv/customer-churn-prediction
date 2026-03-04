import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("customer_churn_model.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer will churn or not.")

st.sidebar.header("Customer Input Features")

# -------- USER INPUT --------

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

monthly_charges = st.sidebar.number_input("Monthly Charges", 10.0, 200.0, 70.0)

total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

support_calls = st.sidebar.slider("Support Calls", 0, 20, 2)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No"]
)

# Feature engineering (same as training)
avg_charge_per_month = total_charges / (tenure + 1)

# Create dataframe
input_data = pd.DataFrame({
    "tenure": [tenure],
    "monthly_charges": [monthly_charges],
    "total_charges": [total_charges],
    "support_calls": [support_calls],
    "avg_charge_per_month": [avg_charge_per_month],
    "contract": [contract],
    "payment_method": [payment_method],
    "internet_service": [internet_service],
    "tech_support": [tech_support],
    "online_security": [online_security]
})

st.subheader("Customer Data")
st.write(input_data)

# -------- PREDICTION --------

if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠ Customer will churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer will stay (Probability of churn: {probability:.2f})")