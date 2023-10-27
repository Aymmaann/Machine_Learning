
# Student Performance Analysis and Prediction: Using Machine Learning

This repository contains code for the analysis of a student performance dataset and the development of a predictive model. The analysis is performed using Python and data science libraries, like Pandas, NumPy, Seaborn, and Matplotlib. The goal is to create a machine learning model to predict student marks based on various features.

## Dataset:

The dataset used in this project includes the following columns:
- `Name`: Student name (not used in the analysis)
- `Age`: Student age
- `Gender`: Student gender
- `Hours_Studied`: Number of hours the student studied
- `Physics_Marks`: Marks obtained in physics
- `Maths_Marks`: Marks obtained in mathematics
- `Chemistry_Marks`: Marks obtained in chemistry
- `Has_Part_Time_Job`: Indicates if the student has a part-time job
- `Study_Hour_Group`: Group indicating the level of study hours

## Exploratory Data Analysis:

The code in this repository performs an extensive Exploratory Data Analysis (EDA) on the student performance dataset. It makes use of various visualisation tools from Seaborn and Matplotlib to visualise univariate, bivariate and multivariate analysis to uncover relationships and correlations of the provided dataset.

## Machine Learning Model

A machine learning model, specifically a Linear Regression model, is trained using the features from the dataset to predict student marks in physics, math, and chemistry. The following steps are executed:

1. Data preprocessing and feature selection.
2. Splitting the data into training (80%) and testing (20%) sets.
3. Model training using the Linear Regression algorithm.
4. Model evaluation and prediction.

## Usage

To predict student marks based on your input, you can run the code and provide values for age, gender, study hours, and IQ. The model will predict the marks in physics, mathematics, and chemistry.

