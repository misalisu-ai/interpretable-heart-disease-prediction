import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from src.preprocessing import preprocess, encode_features
from src.data_loader import load_data # Ensure you can load the CSV

@st.cache_data
def get_background_data():
    """Provides a baseline for SHAP by sampling the training data."""
    df = load_data()
    df = preprocess(df)
    X, _ = encode_features(df)
    # Match the 28 columns
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    # Use the same RFE selection as the model
    X_selected = model.named_steps['rfe'].transform(X)
    # Return a 50-row sample to act as the 'average patient'
    return shap.sample(X_selected, 50)


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

        # 4. Model Explanation (SHAP) 
                
                # 4. Model Explanation (SHAP)
        st.subheader("Model Explanation (SHAP)")
        
        # Get background data for comparison
        background = get_background_data()
        
        # Pull the model parts
        rfe_step = model.named_steps['rfe']
        logreg_model = model.named_steps['logreg']
        selected_features = X_input.columns[rfe_step.support_]

        # Current patient (transformed)
        X_patient_selected = rfe_step.transform(X_input)

        # Create Explainer with the 'background' baseline
        explainer = shap.Explainer(logreg_model, background, feature_names=selected_features)
        shap_values = explainer(X_patient_selected)

        # Explain the predicted class
        class_idx = int(prediction)
        
        fig, ax = plt.subplots()
        # [0, :, class_idx] = 1st patient, all features, specific disease stage
        shap.plots.waterfall(shap_values[0, :, class_idx], show=False)
        
        plt.title(f"Why the model predicted: {prob_df['Class'][class_idx]}")
        st.pyplot(fig)


    except ValueError as e:
        st.error(f"Feature Mismatch Error: {e}")
        st.write("Ensure your encode_features function creates the exact columns used in training.")
