import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
st.set_page_config(page_title="Heart Disease Prediction")
st.title("Heart Disease Predictor")

df=pd.read_csv("heart.csv")
st.write (df.head())
st.write("Dataset Shape: ", df.shape)

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

#REMOVING OUTLIERS
mean=np.mean(df['Age'])
std_dev=np.std(df['Age'])
z_score_age = (df['Age']-mean)/std_dev

mean=np.mean(df['RestingBP'])
std_dev=np.std(df['RestingBP'])
z_score_bp = (df['RestingBP']-mean)/std_dev

mean=np.mean(df['Cholesterol'])
std_dev=np.std(df['Cholesterol'])
z_score_ch = (df['Cholesterol']-mean)/std_dev

mean=np.mean(df['MaxHR'])
std_dev=np.std(df['MaxHR'])
z_score_hr = (df['MaxHR']-mean)/std_dev

no_outliers = (
    (z_score_age < 3) & (z_score_age > -3) &
    (z_score_bp < 3) & (z_score_bp > -3) &
    (z_score_ch < 3) & (z_score_ch > -3) &
    (z_score_hr < 3) & (z_score_hr > -3)
)
filtered_df = df[no_outliers]

# CONVERT LABELS TO NUMBERS
filtered_df['ChestPainType'] = filtered_df['ChestPainType']. map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
filtered_df['Sex'] = filtered_df['Sex']. map({'M':0, 'F': 1})
filtered_df['RestingECG'] = filtered_df['RestingECG']. map({'Normal': 0, 'ST': 1, 'LVH': 2})
filtered_df['ST_Slope'] = filtered_df['ST_Slope']. map({'Up': 0, 'Flat':1, 'Down':2})
filtered_df['ExerciseAngina'] = filtered_df['ExerciseAngina']. map({'N': 0, 'Y': 1})

# FEATURES AND LABELS
x = filtered_df.drop('HeartDisease', axis=1)
y = filtered_df['HeartDisease']

# # FEATURE SCALING AND PCA
scaler=StandardScaler()
x_scaled = scaler.fit_transform(x)
pca = PCA(n_components=8)
x_pca = pca.fit_transform(x_scaled)

# TRAIN TEST SPLIT
x_pca_train, x_pca_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state = 42)

# TRAIN MODEL
model1 = LogisticRegression()
model2 = SVC(kernel='rbf')
model3 = RandomForestClassifier( n_estimators=100, random_state=42)
model1.fit(x_pca_train, y_train)
model2.fit(x_pca_train, y_train)
model3.fit(x_pca_train, y_train)

# MODEL EVALUATION
y_pred1 = model1.predict(x_pca_test)
y_pred2 = model2.predict(x_pca_test)
y_pred3 = model3.predict(x_pca_test) 

accuracy1 = accuracy_score(y_test, y_pred1)
precision1 = precision_score(y_test, y_pred1)
recall1 = recall_score(y_test, y_pred1)
f1_1 = f1_score(y_test, y_pred1)

accuracy2 = accuracy_score(y_test, y_pred2)
precision2 = precision_score(y_test, y_pred2)
recall2 = recall_score(y_test, y_pred2)
f1_2 = f1_score(y_test, y_pred2)

accuracy3 = accuracy_score(y_test, y_pred3)
precision3 = precision_score(y_test, y_pred3)
recall3 = recall_score(y_test, y_pred3)
f1_3 = f1_score(y_test, y_pred3)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM', 'Random Forest'],
    'Accuracy': [accuracy1*100, accuracy2*100, accuracy3*100],
    'Precision': [precision1*100, precision2*100, precision3*100],
    'Recall': [recall1*100, recall2*100, recall3*100],
    'F1 Score': [f1_1*100, f1_2*100, f1_3*100]
})

comparison_df['Accuracy'] = comparison_df['Accuracy'].map(lambda x: f"{x:.2f}")
comparison_df['Precision'] = comparison_df['Precision'].map(lambda x: f"{x:.2f}")
comparison_df['Recall'] = comparison_df['Recall'].map(lambda x: f"{x:.2f}")
comparison_df['F1 Score'] = comparison_df['F1 Score'].map(lambda x: f"{x:.2f}")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.subheader("Model Comparison")
    st.table(comparison_df)

# PREDICT NEW DATA
st.subheader("Predict New Data : ")
age = st.number_input("Age")
sex = st.selectbox("Sex",("M", "F"))
chestPainType= st.selectbox("Chest Pain Type",('ATA', 'NAP', 'ASY', 'TA'))
restingBP = st.number_input("Resting BP")
cholesterol = st.number_input("Cholesterol", )
fastingBS = st.selectbox("Fasting BS",(0, 1))
restingECG = st.selectbox("Resting ECG", ("Normal", "ST", "LVH"))
maxHR = st.number_input("Maximum HR")
exerciseAngina = st.selectbox("Exercise Angina", ("Y", "N"))
oldpeak = st.number_input("Oldpeak")
st_Slope = st.selectbox("ST Slope", ("Up", "Flat", "Down"))

if st.button("Predict Heart Disease"):

    input_data = pd.DataFrame(np.zeros((1, len(x.columns))), columns=x.columns)

    input_data.loc[0, 'Age'] = age
    input_data.loc[0, 'Sex'] = 0 if sex == 'M' else 1
    input_data.loc[0, 'ChestPainType'] = {'ATA':0, 'NAP':1, 'ASY':2, 'TA':3}[chestPainType]
    input_data.loc[0, 'RestingBP'] = restingBP
    input_data.loc[0, 'Cholesterol'] = cholesterol
    input_data.loc[0, 'FastingBS'] = fastingBS
    input_data.loc[0, 'RestingECG'] = {'Normal':0, 'ST':1, 'LVH':2}[restingECG]
    input_data.loc[0, 'MaxHR'] = maxHR
    input_data.loc[0, 'ExerciseAngina'] =  0 if exerciseAngina == 'N' else 1
    input_data.loc[0, 'Oldpeak'] = oldpeak
    input_data.loc[0, 'ST_Slope'] = {'Up':0, 'Flat':1, 'Down':2}[st_Slope]

    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    prediction = model3.predict(input_pca)
    final = prediction[0]
    if final == 1:
        st.error("High Possibility of Heart Disease")
    else:
        st.success("Low Possibility of Heart Disease")