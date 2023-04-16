# Gas Detection and Classification Dataset

This repository contains information about the MultimodalGasData dataset for gas detection and classification. 

## Dataset Information

- Source paper: "MultimodalGasData: Multimodal Dataset for Gas Detection and Classification" 
- DOI: 10.17632/zkwgkjkjn9.2
- Dataset downloaded from: https://data.mendeley.com/datasets/zkwgkjkjn9/2 (accessed on 12/03/23)

## Instrument Used

The dataset was generated using the following instruments:

1. Metal Oxide Semiconductor (MQ) sensors (consisting of 7 sensors, with each one sensitive to certain gases)
2. Seek Compact Thermal Imaging Camera (UW-AAA)

## Model Stages

The dataset can be used for the following stages of machine learning:

1. Basic ML - Only using numerical data to classify the current environmental condition (Decision Tree, kNN)
2. Intermediate ML - Using OpenCV to add thermal camera images as a model feature (increasing accuracy on low concentration gas)
3. Adv ML - Implementing deep learning to create a self-learning model

## Gas Sensors and Sensitive Gases

The following table lists the gas sensors used in the dataset and their corresponding sensitive gases:

| Sensor | Sensitive Gas               |
|--------|-----------------------------|
| MQ2    | LPG, Butane, Methane, Smoke |
| MQ3    | Smoke, Ethanol, Alcohol     |
| MQ5    | LPG, Natural Gas            |
| MQ6    | LPG, Butane                 |
| MQ7    | Carbon Monoxide             |
| MQ8    | Hydrogen                    |
| MQ135  | Air Quality (Smoke, Benzene)|

## Report Analysis
For a detailed analysis of our research report, please refer to the PDF file [Report Analysis.pdf](/Report-Analysis.pdf). This report provides an in-depth discussion of our data analysis, model evaluation, and conclusions. We encourage you to read this report to gain a better understanding of our methodology and results.

## Predict Gas Type API (FastAPI)

### Endpoint

`POST` `/predict_gas`

### Description

This API endpoint accepts sensor values as input and returns the predicted gas type and its probability for both Random Forest and k-Nearest Neighbors models.

### Request

#### Body

| Parameter | Type | Description                           |
|-----------|------|---------------------------------------|
| MQ2       | int  | Sensor value for MQ2                  |
| MQ3       | int  | Sensor value for MQ3                  |
| MQ5       | int  | Sensor value for MQ5                  |
| MQ6       | int  | Sensor value for MQ6                  |
| MQ7       | int  | Sensor value for MQ7                  |
| MQ8       | int  | Sensor value for MQ8                  |
| MQ135     | int  | Sensor value for MQ135                |

#### Example

```json
{
  "MQ2": 2202,
  "MQ3": 799,
  "MQ5": 529,
  "MQ6": 515,
  "MQ7": 507,
  "MQ8": 696,
  "MQ135": 768
}
```

### Response

#### Body
| Parameter        | Type    | Description                                               |
|------------------|---------|-----------------------------------------------------------|
| rf_prediction    | int     | Predicted gas type using Random Forest model              |
| rf_probability   | float   | Probability of the predicted gas type (Random Forest)     |
| knn_prediction   | int     | Predicted gas type using k-Nearest Neighbors model        |
| knn_probability  | float   | Probability of the predicted gas type (k-Nearest Neighbors)|

#### Example

```json
{
  "rf_prediction": 'Perfume',
  "rf_probability": 0.78,
  "knn_prediction": 'Perfume',
  "knn_probability": 0.83
}
```


Next steps::
- Create dockerfile
- Create Github Action workflow
- create report ML PROCESS