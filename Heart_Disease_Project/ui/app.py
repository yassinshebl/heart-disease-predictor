import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

model = joblib.load("../models/final_model.pkl")


st.title("❤️ Heart Disease Predictor")
st.markdown("""
Welcome to the **Heart Disease Risk Predictor**!  
This tool helps you **estimate your risk of heart disease** based on your health information.

Simply enter your details below and you'll receive a **risk prediction** (Low or High) powered by a machine learning model trained on real medical data.
""")
st.warning("⚠️ Note: This tool is not a medical diagnosis. Always consult a healthcare professional for medical concerns.")


with st.container():
    st.markdown("### 👤 Basic Information")
    col1, col2, = st.columns(2, gap="large")
    with col1:
        age = st.number_input("Enter your age:", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Select your gender:", ['Male', 'Female'])
        cp = st.selectbox("Chest Pain Type (cp):", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        fbs = st.selectbox("Is your fasting blood sugar greater than 120 mg/dl?", ["Yes", "No"])
        exang = st.selectbox("Do you experience chest pain during physical activity?", ["Yes", "No"])

    with col2:
        trestbps = st.slider("Enter your resting blood pressure (mm Hg):", 80, 200)
        chol = st.slider("Enter your serum cholesterol level in your blood (mg/dl):", 100, 600)
        thalach = st.slider("Enter your maximum heart rate achieved during exercise:", 70, 210)
        oldpeak = st.slider("Enter the ST depression induced by exercise relative to rest (oldpeak):", 0.0, 6.0)
        ca = st.selectbox("Number of major vessels colored by fluoroscopy:", [0, 1, 2, 3])

    st.markdown("### 🧪 ECG & Blood Markers")

    col3, col4 = st.columns(2, gap="large")
    with col3:
        restecg = st.selectbox("Resting electrocardiographic results:", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
        slope = st.selectbox("Slope of the Peak:", ["Upsloping", "Flat", "Downsloping"]) 
    with col4:
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect", "Unknown"]) 

fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0
cp = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp)
restecg = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg)
slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
thal = ["Normal", "Fixed defect", "Reversible defect", "Unknown"].index(thal)

input_dict = {
    'age': age,
    'sex': 1 if sex == 'Male' else 0,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'ca': ca
}

one_hot_columns = [
    'cp_2.0', 'cp_3.0', 'cp_4.0',
    'restecg_1.0', 'restecg_2.0',
    'slope_2.0', 'slope_3.0',
    'thal_6.0', 'thal_7.0'
]
for col in one_hot_columns:
    input_dict[col] = 0

if cp in [2.0, 3.0, 4.0]:
    input_dict[f'cp_{float(cp)}'] = 1
if restecg in [1.0, 2.0]:
    input_dict[f'restecg_{float(restecg)}'] = 1
if slope in [2.0, 3.0]:
    input_dict[f'slope_{float(slope)}'] = 1
if thal in [6.0, 7.0]:
    input_dict[f'thal_{float(thal)}'] = 1

input_data = pd.DataFrame([input_dict])

st.markdown("---")
if st.button("🔍 Predict"):
    result = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]
    if result[0] == 1:
        st.error(f"⚠️ High risk of heart disease! (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Low risk of heart disease. (Probability: {prob:.2%})")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.markdown("---")
with st.expander("📊 View Heart Disease Overview (Sample Data)"):
    st.write("This chart shows how many people in our dataset were diagnosed with heart disease compared to those who were not.")
    
    df = pd.read_csv("../data/heart_disease_cleaned.csv")
    
    fig, ax = plt.subplots()
    sns.countplot(x='target', data=df, hue='target', palette='Set2', ax=ax, legend=False)

    ax.set_title("Heart Disease Diagnosis Count")
    ax.set_xlabel("Diagnosis Result")
    ax.set_ylabel("Number of People")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Heart Disease", "Heart Disease"])

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)} people', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, color='black')

    st.pyplot(fig)