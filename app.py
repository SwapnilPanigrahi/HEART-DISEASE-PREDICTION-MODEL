import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("best_model.pkl")

# Streamlit app
st.title("Heart Disease Prediction App")

st.write("""
This app predicts the likelihood of heart disease based on user input.
Please fill in the details below:
""")

# Input fields for user data
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])

# Predict button
if st.button("Predict"):
    # Prepare the input data
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display result
    if prediction[0] == 1:
        st.success(f"The model predicts that the patient is likely to have heart disease.")
    else:
        st.success(f"The model predicts that the patient is unlikely to have heart disease.")
    
    st.write(f"Prediction Probability: {prediction_proba[0]}")
