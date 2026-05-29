# Heart Disease Prediction System

A machine learning-based classification web application for predicting the likelihood of heart disease using clinical and medical parameters, built with Python and Streamlit.

## Machine Learning Models Used:
### Classification Models
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier

## Input Features
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- ST Slope

## Target Variable
- Heart Disease (0 = No Disease, 1 = Disease)

## Data Preprocessing
- Removed duplicate records.
- Handled missing values.
- Removed outliers using the Z-score method.
- Applied Label Encoding to categorical features.
- Applied StandardScaler for feature scaling.
- Applied PCA for dimensionality reduction.

## Application Features
- Comparative analysis of multiple machine learning models.
- Real-time heart disease prediction using Streamlit.
- Performance evaluation using multiple classification metrics.

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

## Use Cases
- Assists healthcare professionals in heart disease risk assessment.
- Supports early identification of individuals who may require further medical evaluation.
- Helps medical representatives demonstrate the role of machine learning in healthcare.

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Dataset
Heart Disease Dataset from Kaggle.


**Dataset Link:**  
https://www.kaggle.com/fedesoriano/heart-failure-prediction

## Model Performance
- Logistic Regression Accuracy: 86.26%
- SVM Accuracy: 86.81%
- Random Forest Accuracy: 87.91%

## Run Locally

```bash
pip install -r requirements.txt
py -m streamlit run app.py
