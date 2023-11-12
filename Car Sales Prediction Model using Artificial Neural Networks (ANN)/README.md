
# Car Sales Prediction Model using Artificial Neural Networks (ANN)

## Overview

This project implements a car sales prediction model using an Artificial Neural Network (ANN). The model is trained on a dataset containing information about customers, including their age, annual salary, credit card debt, net worth, and the actual car purchase amount.

## Dataset

The dataset used for training and testing the model includes the following columns:

- Customer Name
- Customer Email
- Country
- Gender
- Age
- Annual Salary
- Credit Card Debt
- Net Worth
- Car Purchase Amount

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow

## Installation

1. Clone the repository:

    ```bash
    git clone [repository-url]
    cd car-sales-prediction-ann
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the `car_sales_prediction_ann.py` script to train the model and make predictions:

    ```bash
    python car_sales_prediction_ann.py
    ```

2. The script will load the dataset, preprocess the data, train the ANN model, and make a sample prediction.

## Model Architecture

The ANN model architecture consists of three layers:

1. Input layer with 5 neurons (corresponding to the features: age, annual salary, credit card debt, net worth, and country).
2. Two hidden layers with 25 neurons each, using the ReLU activation function.
3. Output layer with 1 neuron and linear activation function.

## Training

The model is trained using the Adam optimizer and mean squared error loss function. The training process is visualized using a plot showing the progression of training and validation loss over epochs.

## Sample Prediction

A sample prediction is made using a test input: `[1, 35, 57000, 13000, 250000]`. The expected purchase amount is printed to the console.

Feel free to customize the input data or explore the script for further analysis.




```
## Issues
If you encounter any issues or have suggestions for improvements, please create an issue on GitHub.

## Changelog
Feel free to customize this template to match the specifics of your code and project. It's important to provide clear and concise information that helps users understand how to use your code effectively.