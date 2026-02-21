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
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Screening Tool", "AI Research Analytics"])

# --- PAGE 1: SCREENING TOOL ---
if page == "Screening Tool":
    st.title("🩺 Diabetes Risk Screening Tool")
    st.subheader("Targeting the 'Thin-Fat' South Asian Phenotype")
    
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

    if submit:
        # Prediction Logic (Mapping)
        gender_enc = 1 if gender == "Male" else 0
        activity_enc = 1 if activity == "Yes" else 0
        sitting_map = {"Less than 4 hours": 0, "4–8 hours": 1, "More than 8 hours": 2}
        sitting_enc = sitting_map[sitting]
        
        bmi = weight / ((height / 100) ** 2)
        waist_cm = waist_inches * 2.54
        whtr = waist_cm / height
        
        features = [[age, gender_enc, bmi, waist_cm, whtr, sitting_enc, activity_enc]]
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]
        
        st.divider()
        if prediction[0] == 1:
            st.error(f"### High Risk Identified ({probability:.1%})")
        else:
            st.success(f"### Low Risk Identified ({probability:.1%})")
        
        st.info(f"**Metrics:** BMI: {bmi:.1f} | WHtR: {whtr:.2f}")
        st.write("👈 Use the sidebar to see the **AI Research Analytics** for this model.")

# --- PAGE 2: RESEARCH ANALYTICS ---
elif page == "AI Research Analytics":
    st.title("📊 AI Model Research Analytics")
    st.write("This page explains the quantitative logic behind the Random Forest model.")
    
    st.divider()
    st.subheader("Feature Importance (Variable Weighting)")
    st.write("According to the trained model, these are the factors that most heavily influence a Diabetes Risk prediction in South Asian professionals.")

    # Get feature importance
    importances = model.feature_importances_
    feature_names = ['Age', 'Gender', 'BMI', 'Waist (cm)', 'WHtR', 'Sitting Hours', 'Physical Activity']
    
    importance_df = pd.DataFrame({
        'Factor': feature_names,
        'Importance Weight': importances
    }).sort_values(by='Importance Weight', ascending=True)

    # Display Bar Chart
    st.bar_chart(data=importance_df, x='Factor', y='Importance Weight', horizontal=True)

    

    st.info("""
    **Research Insight:** Note how **WHtR** and **Sitting Hours** often rank higher than BMI. 
    This supports the 'Thin-Fat' phenotype theory where central adiposity is a better predictor than total body weight.
    """)