Heart Disease Prediction System

A machine learning-based web application for predicting the likelihood of heart disease using clinical and medical parameters from a 900+ row Kaggle medical dataset.

Features
- Data cleaning and preprocessing
- Handling missing values and duplicate records
- Outlier removal using Z-score method
- Feature scaling using StandardScaler
- Dimensionality reduction using PCA
- Comparative analysis of multiple ML models
- Real-time heart disease prediction using Streamlit

Machine Learning Models Used
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier

Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

Dataset
Heart Disease Dataset containing clinical attributes such as age, cholesterol, resting blood pressure, ECG results, chest pain type, maximum heart rate, and exercise-induced angina.

Model Performance
- Logistic Regression Accuracy: 86.26%
- SVM Accuracy: 86.81%
- Random Forest Accuracy: 87.91%

Run Locally

```bash
pip install -r requirements.txt
py -m streamlit run app.py
