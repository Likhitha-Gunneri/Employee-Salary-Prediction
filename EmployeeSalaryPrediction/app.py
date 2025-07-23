# 💼 Employee Salary Predictor - Final Version with Fixes

import streamlit as st
import pandas as pd
import joblib

# === Load model, encoders, and feature list ===
model = joblib.load("model/salary_model.pkl")
encoders = joblib.load("model/encoders.pkl")
features = joblib.load("model/features.pkl")

# === Streamlit Page Setup ===
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# === Custom CSS Styling ===
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #74ebd5, #ACB6E5);
    color: #000;
    font-family: 'Segoe UI', sans-serif;
}

[data-testid="stForm"] {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 2rem;
    border-radius: 1rem;
    box-shadow: 0 0 15px rgba(0,0,0,0.15);
    margin-top: 20px;
}

h1 {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 10px;
}

label {
    font-weight: bold !important;
    color: #1a1a1a !important;
    font-size: 16px !important;
}

input, select, textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
}

.stNumberInput input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 5px;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 5px;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 25px;
    transition: 0.3s;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #45a049;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# === App Title ===
st.title("💼 Employee Salary Predictor")
st.markdown("Enter the employee details below to predict if salary is **>50K** or **<=50K**.")

# === Auto-filled Fields ===
auto_fill = {
    'fnlwgt': 100000,
    'capital-gain': 0,
    'capital-loss': 0,
    'education-num': 10,  # This will not be shown to user
    'relationship': 'Not-in-family',
    'race': 'White',
    'marital-status': 'Never-married'
}

# === Input Form ===
with st.form("prediction_form"):
    user_input = {}

    for feature in features:
        if feature == "education-num":
            user_input[feature] = auto_fill[feature]  # auto-fill, no input shown
        elif feature in auto_fill:
            user_input[feature] = auto_fill[feature]
        elif feature in encoders:
            options = encoders[feature].classes_.tolist()
            user_input[feature] = st.selectbox(f"{feature.replace('-', ' ').capitalize()}", options)
        else:
            user_input[feature] = st.number_input(f"{feature.replace('-', ' ').capitalize()}", min_value=0, step=1)

    submitted = st.form_submit_button("🔍 Predict")

# === Prediction Output ===
if submitted:
    try:
        input_df = pd.DataFrame([user_input])

        # Encode categorical columns
        for col in encoders:
            if col in input_df.columns:
                input_df[col] = encoders[col].transform(input_df[col])

        # Make prediction
        pred = model.predict(input_df)[0]
        label = ">50K" if pred == 1 else "<=50K"

        st.markdown("## 📊 Prediction Result")
        if label == ">50K":
            st.markdown(f'''
                <div style="background-color:#d4edda; color:#155724; padding: 20px; border-radius:10px; font-size:18px; text-align:center;">
                    💰 <b>Predicted Income Class: {label}</b>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <div style="background-color:#f8d7da; color:#721c24; padding: 20px; border-radius:10px; font-size:18px; text-align:center;">
                    💼 <b>Predicted Income Class: {label}</b>
                </div>
            ''', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
