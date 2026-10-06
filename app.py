import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Smart Student Grade Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load the trained machine learning pipeline safely using joblib
@st.cache_resource
def load_model():
    pipeline = joblib.load('student_grade_pipeline.pkl')
    return pipeline

pipeline = load_model()

# App Title & Description
st.title("🎓 Smart Student Grade Predictor & Analytics System")
st.markdown("""
Welcome to the interactive **Student Performance Dashboard**. Adjust the student habit parameters and academic features below to predict the expected final grade (**G3**) and get targeted recommendations.
""")

st.sidebar.header("Navigation & Settings")
app_mode = st.sidebar.selectbox("Choose View", ["Student Performance Predictor", "Educator At-Risk Dashboard"])

if app_mode == "Student Performance Predictor":
    st.subheader("📝 Student Habit & Background Parameters")

    # Interactive User Inputs (Sliders & Selectors)
    studytime = st.slider("Weekly Study Time (1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs)", 1, 4, 2)
    failures = st.slider("Number of Past Class Failures", 0, 4, 0)
    absences = st.slider("Number of School Absences", 0, 93, 4)
    health = st.slider("Current Health Status (1: Very Bad to 5: Very Good)", 1, 5, 3)
    freetime = st.slider("Free Time After School (1: Very Low to 5: Very High)", 1, 5, 3)
    goout = st.slider("Going Out with Friends (1: Very Low to 5: Very High)", 1, 5, 3)
    
    higher = st.selectbox("Wants to Take Higher Education?", ["yes", "no"], index=0)
    internet = st.selectbox("Internet Access at Home?", ["yes", "no"], index=0)
    romantic = st.selectbox("In a Romantic Relationship?", ["yes", "no"], index=1)

    # Dynamic sliders for past grades
    g1 = st.slider("First Period Grade (G1)", 0, 20, 10)
    g2 = st.slider("Second Period Grade (G2)", 0, 20, 10)

    # Input DataFrame using all your slider variables
    input_data = pd.DataFrame({
        'school': ['GP'],
        'sex': ['F'],
        'age': [16],
        'address': ['U'],
        'famsize': ['GT3'],
        'Pstatus': ['T'],
        'medu': [2],
        'fedu': [2],
        'mjob': ['other'],
        'fjob': ['other'],
        'reason': ['course'],
        'guardian': ['mother'],
        'traveltime': [1],
        'studytime': [studytime],
        'failures': [failures],
        'schoolsup': ['no'],
        'famsup': ['yes'],
        'paid': ['no'],
        'activities': ['yes'],
        'nursery': ['yes'],
        'higher': [higher],
        'internet': [internet],
        'romantic': [romantic],
        'famrel': [4],
        'freetime': [freetime],
        'goout': [goout],
        'dalc': [1],
        'walc': [1],
        'health': [health],
        'absences': [absences],
        'G1': [g1],
        'G2': [g2]
    })
    if st.button("Predict My Final Grade"):
        pred_score = pipeline.predict(input_data)[0]
        st.success(f"Estimated Final Grade: {pred_score:.2f} / 20")

        # Personalized study advice based on prediction logic
        if pred_score < 10:
            st.error("⚠️ **Status: At-Risk.** Recommendation: Consider increasing weekly study hours, reducing absences, and seeking academic support from instructors.")
        elif pred_score < 14:
            st.warning("⚠️ **Status: Moderate Performance.** Good job, but consistent study routines can push you into the top tier!")
        else:
            st.info("🌟 **Status: Excellent Performance!** Keep up your current habits and study schedule.")

elif app_mode == "Educator At-Risk Dashboard":
    st.subheader("📊 Educator Portal: At-Risk Student Analytics")
    st.markdown("""
    This section is designed for academic advisors and educators to batch-monitor student risk metrics, analyze failure correlation trends, and intervene early.
    """)
    
    # Load dataset for aggregate analytics view if available
    try:
        df_display = pd.read_csv('student_data.csv')
        st.write("### Overview of Student Dataset Summary")
        st.dataframe(df_display.describe())
    except FileNotFoundError:
        st.info("Dataset file not found in repository root. Upload `student_data.csv` to enable aggregate cohort analytics.")
