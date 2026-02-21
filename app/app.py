import streamlit as st
import pickle
import pandas as pd
import os

# 1. Page Configuration
st.set_page_config(page_title="South Asian Diabetes Risk Predictor", layout="centered")

# --- ROBUST PATH LOGIC ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
model_path = os.path.join(root_dir, 'models', 'random_forest_model.pkl')

@st.cache_resource 
def load_model():
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        return None

model = load_model()

if model is None:
    st.error("⚠️ Model file not found!")
    st.info(f"The app is looking for the model at: {model_path}")
    st.write("Please run `python src/train_model.py` first to generate the brain of the AI.")
    st.stop()

# 2. UI Header
st.title("🩺 Diabetes Risk Screening Tool")
st.subheader("Targeting the 'Thin-Fat' South Asian Phenotype")
st.write("This tool uses Machine Learning to predict risk based on Sri Lankan professional lifestyle markers.")

# 3. Input Form - THIS IS THE SECTION THAT WAS MISSING
with st.form("prediction_form"):
    st.header("Personal & Anthropometric Data")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=120, max_value=220, value=170)
    
    with col2:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        waist_inches = st.number_input("Waist Circumference (Inches)", min_value=20, max_value=60, value=32)
        
    st.header("Lifestyle Factors")
    sitting = st.selectbox("Daily Sitting Hours", ["Less than 4 hours", "4–8 hours", "More than 8 hours"])
    activity = st.radio("Do you engage in 150min of exercise per week?", ["Yes", "No"])
    
    submit = st.form_submit_button("Analyze Risk Profile")

# 4. Logic for Prediction
if submit:
    # Convert words to numbers (Mapping)
    gender_enc = 1 if gender == "Male" else 0
    activity_enc = 1 if activity == "Yes" else 0
    sitting_map = {"Less than 4 hours": 0, "4–8 hours": 1, "More than 8 hours": 2}
    sitting_enc = sitting_map[sitting]
    
    # Calculate Phenotype Metrics
    waist_cm = waist_inches * 2.54
    bmi = weight / ((height / 100) ** 2)
    whtr = waist_cm / height
    
    # Feature Array (Must match training order)
    features = [[age, gender_enc, bmi, waist_cm, whtr, sitting_enc, activity_enc]]
    
    # Prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]
    
    # 5. Display Result
    st.divider()
    if prediction[0] == 1:
        st.error(f"### High Risk Identified")
        st.write(f"Probability of metabolic risk: **{probability:.1%}%**")
    else:
        st.success(f"### Low Risk Identified")
        st.write(f"Probability of metabolic risk: **{probability:.1%}%**")

    # Metrics Section
    st.info(f"**Calculated Metrics:** BMI: {bmi:.1f} | WHtR: {whtr:.2f}")
    if whtr > 0.5:
        st.warning("⚠️ Your Waist-to-Height Ratio is above 0.5. This is a significant risk marker for South Asians.")