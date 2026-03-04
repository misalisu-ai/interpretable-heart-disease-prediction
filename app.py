import streamlit as st
import pandas as pd
import joblib

st.title("Interpretable Heart Disease Risk Classifier")

# Load model
model = joblib.load("models/heart_disease_model.pkl")

st.header("Patient Input")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])
cp = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl (0/1)", [0,1])
restecg = st.selectbox("Resting ECG (0–2)", [0,1,2])
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (0/1)", [0,1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope (0–2)", [0,1,2])
ca = st.selectbox("Major Vessels (0–4)", [0,1,2,3,4])
thal = st.selectbox("Thal (0–3)", [0,1,2,3])

input_dict = {
    'age':[age],
    'sex':[sex],
    'cp':[cp],
    'trestbps':[trestbps],
    'chol':[chol],
    'fbs':[fbs],
    'restecg':[restecg],
    'thalach':[thalach],
    'exang':[exang],
    'oldpeak':[oldpeak],
    'slope':[slope],
    'ca':[ca],
    'thal':[thal]
}

input_df = pd.DataFrame(input_dict)

if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.subheader("Prediction")

    if prediction == 0:
        st.success("No Disease")
    elif prediction == 1:
        st.warning("Mild Disease")
    else:
        st.error("Severe Disease")

    st.subheader("Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Class": ["No Disease", "Mild Disease", "Severe Disease"],
        "Probability": probabilities
    })

    st.bar_chart(prob_df.set_index("Class"))
