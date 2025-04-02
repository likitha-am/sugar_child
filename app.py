import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("diabetes_model.pkl")

st.title("Diabetes Prediction App")

# Collect user input
preg = st.number_input("Number of Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Predict button
if st.button("Predict"):
    features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("The person is likely to have diabetes.")
    else:
        st.success("The person is not likely to have diabetes.")
