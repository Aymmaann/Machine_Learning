
# Stroke Prediction Model 
This repository contains a script for building a stroke prediction model using TensorFlow and Keras. The model is designed for binary classification to predict whether an individual is likely to have a stroke based on various features. Below is a breakdown of the script and its functionalities.

## Table of Contents
1. Introduction
2. Data Loading and Exploration
3. Data Preprocessing
4. Model Building
5. Training and Evaluation
6. Results and Visualizations
7. Conclusion

## Introduction
The goal of this script is to create a machine learning model that predicts the likelihood of an individual having a stroke based on various features. The dataset used for training and evaluation is loaded from a CSV file ('stroke.csv').

## Data Loading and Exploration
- The script begins by loading the stroke dataset using the pandas library.
- Exploratory data analysis (EDA) is conducted using methods such as head(), shape, columns, describe(), and info() to understand the dataset's structure and characteristics.
- Skewness of the 'bmi' column is calculated to assess its distribution.

## Data Preprocessing
- Missing values in the 'bmi' column are filled with the median, considering its skewness.
- The 'id' column is dropped as it is likely an identifier and may not contribute to the prediction task.
- Categorical variables ('gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status') are encoded using LabelEncoder from scikit-learn.


## Model Building
- A simple neural network model is defined using TensorFlow and Keras.
- The model consists of an input layer with the same dimensionality as the number of features and an output layer with one neuron and a sigmoid activation function for binary classification.


## Training and Evaluation
- The dataset is split into training and testing sets using train_test_split from scikit-learn.
- Feature scaling is performed using StandardScaler to standardize the feature values.
- The model is compiled with binary crossentropy loss and accuracy as the metric.
- Training is conducted for 100 epochs, and the model's performance is evaluated on both the training and testing sets.

## Results and Visualizations
- Loss and accuracy curves are plotted for both training and validation sets to visualize the model's performance over epochs.
- The final training and testing scores are printed to assess the model's accuracy.

## Conclusion
This script provides a comprehensive workflow for building a stroke prediction model, from data loading and preprocessing to model training and evaluation. Users can customize hyperparameters, experiment with different architectures, and further optimize the model based on their specific use case.

*Adjustments can be made to the model architecture, hyperparameters, and data preprocessing steps to improve performance and adapt to different datasets.*