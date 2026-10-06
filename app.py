import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# Page Configuration
st.set_page_config(page_title="Smart Student Grade Predictor", page_icon="🎓", layout="wide")

# Load Trained Pipeline
@st.cache_resource
def load_model():
    return joblib.load('student_grade_pipeline.pkl')

pipeline = load_model()

# App Header
st.title("🎓 Smart Student Grade Predictor & Intervention System")
st.markdown("A predictive machine learning system with **Explainable AI (SHAP)** designed for early academic interventions.")

# Sidebar Navigation
app_mode = st.sidebar.selectbox("Choose Portal Mode", ["Student Self-Service View", "Educator At-Risk Dashboard"])

# ----------------------------------------------------
# 1. STUDENT VIEW PORTAL
# ----------------------------------------------------
if app_mode == "Student Self-Service View":
    st.header("📖 Student Self-Service Portal")
    st.markdown("Enter your current habits and routines to estimate your final academic performance and receive tailored advice.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        studytime = st.selectbox("Weekly Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)", [1, 2, 3, 4], index=1)
        absences = st.number_input("Number of Absences", min_value=0, max_value=93, value=4)
        failures = st.selectbox("Past Class Failures", [0, 1, 2, 3], index=0)
    
    with col2:
        health = st.slider("Health Status (1: Very Bad - 5: Very Good)", 1, 5, 3)
        freetime = st.slider("Free Time After School (1: Very Low - 5: Very High)", 1, 5, 3)
        goout = st.slider("Going Out with Friends (1: Very Low - 5: Very High)", 1, 5, 3)
    
    with col3:
        higher = st.selectbox("Wants Higher Education?", ["yes", "no"], index=0)
        internet = st.selectbox("Internet Access at Home?", ["yes", "no"], index=0)
        romantic = st.selectbox("In a Romantic Relationship?", ["yes", "no"], index=1)

    # Read original dataset to satisfy all pipeline columns
# Create a neutral baseline dictionary with default values for all columns
input_data = pd.DataFrame({
    'school': ['GP'],
    'sex': ['F'],
    'age': [16],
    'address': ['U'],
    'famsize': ['GT3'],
    'Pstatus': ['T'],
    'Medu': [2],
    'Fedu': [2],
    'Mjob': ['other'],
    'Fjob': ['other'],
    'reason': ['course'],
    'guardian': ['mother'],
    'traveltime': [1],
    'studytime': [studytime],      # Controlled by slider
    'failures': [failures],          # Controlled by slider
    'schoolsup': ['no'],
    'famsup': ['yes'],
    'paid': ['no'],
    'activities': ['yes'],
    'nursery': ['yes'],
    'higher': [higher],              # Controlled by selector
    'internet': [internet],          # Controlled by selector
    'romantic': [romantic],          # Controlled by selector
    'famrel': [4],
    'freetime': [freetime],          # Controlled by slider
    'goout': [goout],                # Controlled by slider
    'Dalc': [1],
    'Walc': [1],
    'health': [health],              # Controlled by slider
    'absences': [absences],          # Controlled by slider
    'G1': [10],                      # Neutral default mid-grade
    'G2': [10]                       # Neutral default mid-grade
})
if st.button("Predict My Final Grade"):
    pred_score = pipeline.predict(input_data)[0]
    st.success(f"### Predicted Final Grade (G3): **{pred_score:.1f} / 20**")
    
   
        # Personalized study advice based on prediction
        if pred_score < 10:
            st.error("⚠️ **Status: At-Risk.** Recommendation: Consider increasing weekly study hours, reducing absences, and seeking academic support from instructors.")
        elif pred_score < 14:
            st.warning("⚡ **Status: Moderate Performance.** Recommendation: Good job, but consistent study routines can push you into the top tier!")
        else:
            st.info("🌟 **Status: Excellent Performance!** Keep up your current habits and study schedule.")

# ----------------------------------------------------
# 2. EDUCATOR DASHBOARD PORTAL
# ----------------------------------------------------
elif app_mode == "Educator At-Risk Dashboard":
    st.header("🏫 Educator At-Risk Monitoring Dashboard")
    st.markdown("Upload bulk student CSV records to automatically flag students who need early intervention.")
    
    uploaded_file = st.file_uploader("Upload Student CSV File", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Successfully loaded **{batch_df.shape[0]}** student records.")
        
        # Make predictions
        if 'G3' in batch_df.columns:
            features_to_predict = batch_df.drop(columns=['G3'])
        else:
            features_to_predict = batch_df.copy()
            
        batch_preds = pipeline.predict(features_to_predict)
        batch_df['Predicted_G3'] = batch_preds
        
        # Flag at-risk students (e.g., Predicted G3 < 10)
        at_risk_df = batch_df[batch_df['Predicted_G3'] < 10]
        
        st.subheader(f"🚨 At-Risk Students Flagged: {len(at_risk_df)} out of {len(batch_df)}")
        
        if len(at_risk_df) > 0:
            st.dataframe(at_risk_df[['age', 'absences', 'studytime', 'failures', 'Predicted_G3']])
            
            # Download button for at-risk report
            csv = at_risk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download At-Risk Student Report (CSV)",
                data=csv,
                file_name='at_risk_students_report.csv',
                mime='text/csv',
            )
        else:
            st.success("No students are currently flagged as at-risk based on the threshold!")
    else:
        st.info("Tip: You can upload your original `student_data.csv` file here to test the bulk dashboard view.")