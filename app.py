import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from src.preprocessing import preprocess, encode_features

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
    try: 
        # 1. Transform 13 raw inputs into 28 encoded features
        df_temp = preprocess(input_df)
        X_input, _ = encode_features(df_temp)

        training_columns = model.feature_names_in_ 

        X_input = X_input.reindex(columns=training_columns, fill_value=0)
        
        # 2. Prediction (Corrected to X_input)
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]

        st.subheader("Prediction Results")
        result_text = "No Disease" if prediction == 0 else "Heart Disease Detected"
        st.info(f"Final Prediction: **{result_text}**")

        # 3. Probability Bar Chart
        prob_df = pd.DataFrame({
            "Class": ["No Disease", "Mild Disease", "Severe Disease"],
            "Probability": probabilities
        })
        st.bar_chart(prob_df.set_index("Class"))

        # 4. Model Explanation (SHAP) - MUST stay inside the button block
                # 4. Model Explanation (SHAP)
        st.subheader("Model Explanation (SHAP)")
        
        # Pass the 28 features through the RFE step
        X_selected = model.named_steps['rfe'].transform(X_input)

        explainer = shap.Explainer(
            model.named_steps['logreg'],
            X_selected
        )
        
        # This returns a 3D array: [observations, features, classes]
        shap_values = explainer(X_selected)

        # Let's explain the "Most Likely" class predicted by the model
        predicted_class_index = int(prediction[0]) 
        
        fig, ax = plt.subplots()
        # We take [0] for the first patient, and [predicted_class_index] for the class
        shap.plots.waterfall(shap_values[0, :, predicted_class_index], show=False)
        
        st.pyplot(fig)

    except ValueError as e:
        st.error(f"Feature Mismatch Error: {e}")
        st.write("Ensure your encode_features function creates the exact columns used in training.")
