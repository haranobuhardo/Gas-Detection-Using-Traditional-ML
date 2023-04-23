import requests
from datetime import datetime as dt

# FastAPI endpoint URL
FASTAPI_URL = "http://127.0.0.1:8000/predict_gas"

def test_prediction(): 
    """
    Simulating an ideal input condition
    Expecting a 200 response with proper JSON data feedback
    """
    
    # Sample sensor data
    sensor_data = {
        "MQ2": 1000,
        "MQ3": 300,
        "MQ5": 300,
        "MQ6": 300,
        "MQ7": 500,
        "MQ8": 800,
        "MQ135": 332
    }

    # Send a POST request to the FastAPI endpoint with sensor values
    # and record time elapsed
    start_time = dt.now()
    response = requests.post(FASTAPI_URL, json=sensor_data)
    finished_time = dt.now()
    elapsed_time = finished_time - start_time
    elapsed_time = elapsed_time.total_seconds()

    # Check if the request was successful
    if response.status_code == 200:
        # Extract the predicted gas type and probabilities
        predictions = response.json()

        # Display the results
        # print(f"Random Forest Prediction: {predictions['rf_prediction']} (Probability: {predictions['rf_probability']:.2f})")
        print(f"k-Nearest Neighbors Prediction: {predictions['knn_prediction']} (Probability: {predictions['knn_probability']:.2f})")
    else:
        print("Error: Unable to get predictions. Please check the FastAPI server.")
    print("Time elapsed: {:.5f} seconds".format(elapsed_time))

def test_out_of_range(): 
    """
    Simulating an input with one sensor out of range value
    Expecting a 400 error response
    """

    # Sample sensor data
    sensor_data = {
        "MQ2": 200,
        "MQ3": 300,
        "MQ5": 300,
        "MQ6": 300,
        "MQ7": 500,
        "MQ8": 1200,
        "MQ135": 332
    }

    # Send a POST request to the FastAPI endpoint with sensor values
    # and record time elapsed
    start_time = dt.now()
    response = requests.post(FASTAPI_URL, json=sensor_data)
    finished_time = dt.now()
    elapsed_time = finished_time - start_time
    elapsed_time = elapsed_time.total_seconds()

    # Check if the request was successful
    if response.status_code == 422:
        print("422 Unprocessable Entity Error response received.")
    else:
        print("Error: Unable to get predictions. Please check the FastAPI server.")
    print("Time elapsed: {:.5f} seconds".format(elapsed_time))

def test_mistype_input(): 
    """
    Simulating an input with mistype (expecting int, but input with str)
    Expecting a 400 response
    """

    # Sample sensor data
    sensor_data = {
        "MQ2": 100,
        "MQ3": '1o0',
        "MQ5": 300,
        "MQ6": 300,
        "MQ7": 500,
        "MQ8": 800,
        "MQ135": 332
    }

    # Send a POST request to the FastAPI endpoint with sensor values
    # and record time elapsed
    start_time = dt.now()
    response = requests.post(FASTAPI_URL, json=sensor_data)
    finished_time = dt.now()
    elapsed_time = finished_time - start_time
    elapsed_time = elapsed_time.total_seconds()

    # Check if the request was successful
    if response.status_code == 422:
        print("422 Unprocessable Entity Error response received.")
    else:
        print("Error: Unable to get predictions. Please check the FastAPI server.")
        print(response.json())
    print("Time elapsed: {:.5f} seconds".format(elapsed_time))


if __name__=='__main__':
    print("Test 1: Ideal input")
    test_prediction()
    print("Test 2: Out of range input")
    test_out_of_range()
    print("Test 3: Mistype input")
    test_mistype_input()