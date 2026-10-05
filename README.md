# 🎓 Smart Student Grade Predictor System

An interactive machine learning web application designed for early academic intervention. This system predicts a student's final academic performance (G3) based on behavioral, social, and demographic habits, while providing transparent insights using Explainable AI (SHAP).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL_HERE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Live Demo
Experience the interactive web application live on Streamlit Cloud:  
👉 **[Click here to view the Live App](YOUR_STREAMLIT_URL_HERE)**

---

## 🚀 Project Overview
Educational institutions often struggle to identify at-risk students before it is too late. This project bridges that gap by leveraging predictive modeling and machine learning to estimate student grades and highlight key risk factors. 

Key features include:
* **Predictive Modeling:** Utilizes a trained machine learning pipeline to forecast student final grades.
* **Explainable AI (SHAP):** Breaks down individual predictions so educators and students understand *why* a score was predicted (e.g., impact of study time vs. absences).
* **Interactive Self-Service Portal:** Allows users to adjust habit sliders and lifestyle metrics in real-time to see potential outcomes.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Processing & Manipulation:** Pandas, NumPy
* **Machine Learning & Modeling:** Scikit-Learn, CatBoost, Joblib
* **Model Interpretability:** SHAP (SHapley Additive exPlanations)
* **Web Dashboard & Deployment:** Streamlit, Streamlit Community Cloud
* **Version Control:** Git & GitHub

---

## 📂 Repository Structure
```text
student-grade-predictor-system/
│
├── app.py                      # Main Streamlit web application
├── student_grade_pipeline.pkl  # Trained machine learning model pipeline
├── student_data.csv            # Dataset used for modeling/reference
├── Student_grade_predictor.ipynb # Jupyter Notebook containing EDA & training
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

Key Highlights & Insights
Analyzed behavioral attributes such as weekly study hours, past class failures, absences, and social habits.

Built robust data preprocessing pipelines to handle feature scaling and categorical encoding smoothly.

Ensured seamless end-to-end integration from raw exploratory data analysis (EDA) in Jupyter Notebooks to production deployment.
Live Dashboard link
https://student-grade-predictor-system-pmdrxvgw2fjayurfzzfepv.streamlit.app/

AUTHOR
Kanika Joshi
