import streamlit as st
import pandas as pd
import pickle


# Load trained model pipeline
with open("disease_diagnosis_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Disease Diagnosis Prediction", layout="wide")

# App Title
st.title("🧑‍⚕️ Disease Diagnosis Prediction App")
st.write("Enter patient details below to predict the disease diagnosis.")

# Predefine categories (from your dataset)
# ---------------------------
gender_options = ["Male", "Female"]

# You can replace these with `data["Symptom_1"].unique()` etc. if you load dataset
symptom_1_options = ['Fatigue', 'Sore throat', 'Body ache', 'Shortness of breath','Runny nose', 'Headache', 'Cough', 'Fever']  
symptom_2_options = ['Fatigue', 'Sore throat', 'Body ache', 'Shortness of breath','Runny nose', 'Headache', 'Cough', 'Fever']  
symptom_3_options = ['Fatigue', 'Sore throat', 'Body ache', 'Shortness of breath','Runny nose', 'Headache', 'Cough', 'Fever']  
severity_options = ["Mild", "Moderate", "Severe"]  
treatment_options = ['Medication and rest', 'Rest and fluids','Hospitalization and medication'] 
# Input form
with st.form("patient_form"):

    # Numerical inputs
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=75)
    body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
    oxygen_sat = st.number_input("Oxygen Saturation (%)", min_value=50, max_value=100, value=98)
    systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
    diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=140, value=80)

    # Categorical inputs
    gender = st.selectbox("Gender", gender_options)
    symptom_1 = st.selectbox("Symptom 1", symptom_1_options)
    symptom_2 = st.selectbox("Symptom 2", symptom_2_options)
    symptom_3 = st.selectbox("Symptom 3", symptom_3_options)
    severity = st.selectbox("Severity", severity_options)
    treatment = st.selectbox("Treatment Plan", treatment_options)

    # Submit button
    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    # Prepare input
    input_data = pd.DataFrame([{
        "Age": age,
        "Heart_Rate_bpm": heart_rate,
        "Body_Temperature_C": body_temp,
        "Oxygen_Saturation_%": oxygen_sat,
        "Systolic_BP": systolic_bp,
        "Diastolic_BP": diastolic_bp,
        "Gender": gender,
        "Symptom_1": symptom_1,
        "Symptom_2": symptom_2,
        "Symptom_3": symptom_3,
        "Severity": severity,
        "Treatment_Plan": treatment
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    # Output result
    st.subheader("Prediction Result")
    st.write(f"**Predicted Disease:** {prediction}")
