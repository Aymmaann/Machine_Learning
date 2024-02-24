# Cancer Diagnosis Prediction
This project focuses on building a neural network model to predict cancer diagnosis based on input features. The dataset used for this project is sourced from 'cancer.csv'.


## Table of Contents
- Introduction
- Dataset Overview
- Dataset Exploration
- Neural Network Architecture
- Usage
- Dependencies


## Introduction
This project aims to predict cancer diagnosis (1 for malignant, 0 for benign) using a neural network. The model is built using TensorFlow and Keras.


# Dataset Overview
The dataset 'cancer.csv' is used for training and testing the model. The data contains various features, and the target variable is 'diagnosis(1=m, 0=b)'.


## Dataset Exploration
Basic information about the dataset is explored using data.info() and data.describe().
Features are extracted and separated into input (X) and target (y) variables.


## Neural Network Architecture
The neural network architecture used for cancer diagnosis prediction consists of three layers:

1. Input Layer:
- Size: The input layer size is determined by the number of features in the dataset (X_train.shape[1:]).
- Activation Function: Sigmoid activation function.

2. Hidden Layer 1:
- Size: 256 neurons.
- Activation Function: Sigmoid activation function.

3. Hidden Layer 2:
- Size: 256 neurons.
- Activation Function: Sigmoid activation function.

4. Output Layer:
- Size: 1 neuron.
- Activation Function: Sigmoid activation function.

5. Model Compilation:
- Optimizer: Adam optimizer.
- Loss Function: Binary crossentropy.
- Metrics: Accuracy.

6. Training:
- The model is trained for 1000 epochs on the training data (X_train and y_train).

This architecture is designed to effectively learn and predict cancer diagnosis based on the provided dataset.


## Usage
- Clone the repository.
- Ensure you have the necessary dependencies installed (see Dependencies).
- Run the Python script cancer_prediction.py to train and evaluate the model.
- View the training progress and evaluation metrics.


## Dependencies
- Python 3
- NumPy
- pandas
- TensorFlow
- scikit-learn