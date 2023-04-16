import requests

# FastAPI endpoint URL
FASTAPI_URL = "http://127.0.0.1:8000/predict_gas"

# Sample sensor data
sensor_data = {
    "MQ2": 500,
    "MQ3": 300,
    "MQ5": 300,
    "MQ6": 300,
    "MQ7": 500,
    "MQ8": 800,
    "MQ135": 332
}

# Send a POST request to the FastAPI endpoint with sensor values
response = requests.post(FASTAPI_URL, json=sensor_data)

# Check if the request was successful
if response.status_code == 200:
    # Extract the predicted gas type and probabilities
    predictions = response.json()

    # Display the results
    print(f"Random Forest Prediction: {predictions['rf_prediction']} (Probability: {predictions['rf_probability']:.2f})")
    print(f"k-Nearest Neighbors Prediction: {predictions['knn_prediction']} (Probability: {predictions['knn_probability']:.2f})")
else:
    print("Error: Unable to get predictions. Please check the FastAPI server.")