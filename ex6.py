import pandas as pd
import streamlit as st

# File uploader for WHO dataset
uploaded_file = st.file_uploader("Upload WHO Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())
else:
    st.write("Please upload a CSV file.")

# Your Bayesian Network code and Streamlit app code would go here
# Assume you already have the BayesianNetwork class and app logic
# For simplicity, let's use the previously defined BayesianNetwork and app code

# Instantiate Bayesian Network
bayesian_network = BayesianNetwork(data)

# Streamlit app
st.title('Corona Infection Diagnosis')

# User input for symptoms
symptoms = {}
for feature in data.columns[:-1]:  # Exclude 'Diagnosis'
    symptoms[feature] = st.selectbox(f'Select {feature}', data[feature].unique())

# Predict diagnosis
if st.button('Diagnose'):
    diagnosis = bayesian_network.predict(symptoms)
    st.write(f'The diagnosis based on the symptoms is: {"Positive" if diagnosis == 1 else "Negative"}')
