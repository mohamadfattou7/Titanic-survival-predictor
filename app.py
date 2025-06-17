import streamlit as st
import pandas as pd
import pickle

# Load the trained model pipeline
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Titanic Survival Prediction")

# Collect user inputs
pclass = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3], index=2)
sex = st.selectbox("Sex", options=["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Passenger Fare", min_value=0.0, value=32.20)
embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q"], index=0)

if st.button("Predict Survival"):
    input_dict = {
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    }
    input_df = pd.DataFrame(input_dict)

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.success(f"Survived 🎉 with probability {prediction_proba:.2f}")
    else:
        st.error(f"Did not survive 😞 with probability {prediction_proba:.2f}")
