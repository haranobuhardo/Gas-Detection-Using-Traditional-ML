import streamlit as st
import requests

# FastAPI endpoint URL
FASTAPI_URL = "http://127.0.0.1:8000/predict_gas"

# Streamlit app title
st.title("Gas Type Prediction")

# Input fields for sensor values
MQ2 = st.slider("MQ2", min_value=0, max_value=1000, value=0)
MQ3 = st.slider("MQ3", min_value=0, max_value=1000, value=0)
MQ5 = st.slider("MQ5", min_value=0, max_value=1000, value=0)
MQ6 = st.slider("MQ6", min_value=0, max_value=1000, value=0)
MQ7 = st.slider("MQ7", min_value=0, max_value=1000, value=500)
MQ8 = st.slider("MQ8", min_value=0, max_value=1000, value=0)
MQ135 = st.slider("MQ135", min_value=0, max_value=1000, value=500)

# Submit button to send a request to the FastAPI endpoint
if st.button("Predict Gas Type"):
    # Package sensor values into a dictionary
    data = {
        "MQ2": MQ2,
        "MQ3": MQ3,
        "MQ5": MQ5,
        "MQ6": MQ6,
        "MQ7": MQ7,
        "MQ8": MQ8,
        "MQ135": MQ135,
    }

    # Send a POST request to the FastAPI endpoint with sensor values
    response = requests.post(FASTAPI_URL, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        # Extract the predicted gas type and probabilities
        predictions = response.json()

        # Display the results
        # st.write(f"Random Forest Prediction: {predictions['rf_prediction']} (Probability: {predictions['rf_probability']:.2f})")
        st.write(f"k-Nearest Neighbors Prediction: {predictions['knn_prediction']} (Probability: {predictions['knn_probability']:.2f})")
    else:
        st.write("Error: Unable to get predictions. Please check the FastAPI server.")
