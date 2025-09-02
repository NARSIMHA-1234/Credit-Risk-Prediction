"""
Streamlit App for Credit Risk Prediction

This app:
1. Loads the trained XGBoost model and preprocessing objects (scaler + encoders)
2. Accepts user input for borrower and loan details
3. Preprocesses the inputs (encoding + scaling)
4. Predicts loan default risk and shows probability

Author: YOUR NAME
"""

import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Saved Model & Preprocessing Objects
# ----------------------------
model = joblib.load("xgboost_model.pkl")       # trained XGBoost model
scaler = joblib.load("scaler.pkl")             # StandardScaler for numeric features
encoders = joblib.load("encoders.pkl")         # LabelEncoders for categorical features
feature_order = joblib.load("feature_order.pkl")  # column order used during training

st.title("📊 Credit Risk Prediction App")
st.write("Enter borrower and loan details to predict the probability of default.")

# ----------------------------
# User Inputs
# ----------------------------
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income", min_value=1000, max_value=500000, value=50000)
person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
loan_amnt = st.number_input("Loan Amount", min_value=500, max_value=50000, value=10000)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=30.0, value=10.0)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.01, max_value=1.0, value=0.2)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)

# Dropdowns for categorical features
home_ownership = st.selectbox("Home Ownership", encoders["person_home_ownership"].classes_)
loan_intent = st.selectbox("Loan Intent", encoders["loan_intent"].classes_)
loan_grade = st.selectbox("Loan Grade", encoders["loan_grade"].classes_)
cb_default = st.selectbox("Default on File", encoders["cb_person_default_on_file"].classes_)

# ----------------------------
# Encode Inputs
# ----------------------------
encoded_inputs = {
    "person_age": person_age,
    "person_income": person_income,
    "person_emp_length": person_emp_length,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "person_home_ownership": encoders["person_home_ownership"].transform([home_ownership])[0],
    "loan_intent": encoders["loan_intent"].transform([loan_intent])[0],
    "loan_grade": encoders["loan_grade"].transform([loan_grade])[0],
    "cb_person_default_on_file": encoders["cb_person_default_on_file"].transform([cb_default])[0],
}

# Put into DataFrame and ensure correct feature order
input_df = pd.DataFrame([encoded_inputs])[feature_order]

# Scale numeric features
input_scaled = scaler.transform(input_df)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict"):
    pred_prob = model.predict_proba(input_scaled)[0][1]   # probability of default
    pred = model.predict(input_scaled)[0]                 # 0 = No Default, 1 = Default

    st.subheader("Prediction Results")
    st.write(f"**Predicted Loan Default:** {'✅ No' if pred == 0 else '⚠️ Yes'}")
    st.write(f"**Probability of Default:** {pred_prob:.2f}")
