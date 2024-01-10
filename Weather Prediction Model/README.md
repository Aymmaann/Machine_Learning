# Weather Prediction Model

## Overview
This folder contains a weather prediction model using multilinear regression based on a dataset named weather.csv. The model predicts the maximum temperature (MaxTemp) based on various weather features. A few columns of the dataset has been listed below:

- 'MinTemp'
- 'MaxTemp'
- 'Rainfall'
- 'Evaporation'
- 'Sunshine'
- 'WindGustDir'
- 'WindGustSpeed'

## Prerequisites
- Python (3.7 or later)
- Jupyter Notebook (optional)
- Install the required packages:

Run the code by:
```bash
python weather_prediction.py
```
(preferably in a jupyter notebook)


## Usage
1. Run the Jupyter Notebook:
```bash
jupyter notebook
```
2. Open the weather.ipynb notebook.

3. Execute the cells to load the dataset, explore the data, preprocess features, and train the Multilinear Regression.

## Features
- Data Exploration: Analyze and visualize the dataset to gain insights into employee attrition patterns.

- Data Preprocessing: Clean and prepare the data, including handling missing values, encoding categorical features, and scaling.

- Model Training: Use Linear Regression to train the model on the preprocessed data.

- Model Evaluation: Evaluate the model's performance using accuracy as the metric.

## Model Evaluation
The model's performance is assessed using the Mean Squared Error, providing an indication of how well the model predicts maximum temperatures. The model achieved an accuracy of 96% on the testing set.

*Feel free to explore and modify the code to enhance the model or experiment with different algorithms.*   