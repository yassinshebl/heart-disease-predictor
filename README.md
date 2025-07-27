# ❤️ Heart Disease Predictor

A Streamlit web application that predicts the likelihood of heart disease based on medical input data. Built with a full machine learning pipeline, including preprocessing, PCA, model selection, and deployment. 

---

## 1. Project Overview

This project utilizes the UCI Heart Disease dataset and includes:

- Data preprocessing  
- Feature selection and dimensionality reduction (PCA)  
- Supervised and unsupervised ML models  
- Model evaluation and tuning  
- Deployment with **Streamlit**  
- Public sharing using **Ngrok**

---

## 2. Models Used

- Logistic Regression  
- K-Nearest Neighbors  
- Random Forest  
- Support Vector Machine  
- KMeans (for unsupervised learning)

---

## 3. Project Structure

```
Heart_Disease_Project/
│── data/
│ ├── heart_disease_cleaned.csv
│ ├── heart_disease_pcs.csv
│ ├── heart_disease_raw.csv
│ ├── X_reduced_with_target.csv
│── deployment/
│ ├── ngrok_setup.txt
│── models/
│ ├── final_model.pkl
│── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ ├── 06_hyperparameter_tuning.ipynb
│── results/
│ ├── evaluation_metrics.txt
│── ui/
│ ├── app.py (Streamlit UI)
│── requirements.txt
```

---

## 3. How to Run

### 3.1 Clone the Repository

```bash
git clone https://github.com/yassinshebl/heart-disease-predictor.git
cd heart-disease-predictor
```

### 3.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📌 Dataset Info

- **Source:** [UCI Machine Learning Repository – Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)  
- Includes features like age, sex, chest pain type, cholesterol, fasting blood sugar, etc.

---

## ⚠️ Disclaimer

> This tool is not a medical diagnosis. Always consult a healthcare professional for medical concerns.

---

## 👤 Author

- **Name:** Yassin Shebl  
- **GitHub:** [@yassinshebl](https://github.com/yassinshebl)
